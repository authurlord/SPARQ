#!/usr/bin/env python3
"""
Context Length Experiment for TableBench
Studies how LLM input text length affects QA performance.

Key Design:
- Truncate tables ONLY for LLM prompts (Router, Check, Final QA)
- SQL execution always uses FULL tables
- Run multiple configurations (8k, 16k, 32k, 64k, 128k tokens)
"""

import os
import sys
import argparse
import json
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import re
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_llm import infer_prompts
from utils.schedule_utils import (
    table_to_str_sql, find_intersection_and_add_row_id,
    format_document, batch_rerank_scores, ROLLBACK,
    merge_clean_and_format_df_dict, retrieve_rows_by_subtables,
    process_error_analysis_list
)
from utils.evaluator import evaluate_tablebench_predictions
from utils.prompt_generate import (
    format_table_prompt, fix_sql_query, filter_dataframe_from_responses,
    match_subtables, retrieve_rows_by_subtables
)
from utils.multi_db_v2 import NeuralDB, Executor
from FlagEmbedding import FlagReranker

import multiprocessing as mp

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Context Length Experiment for TableBench")
    
    # Model paths
    parser.add_argument('--llm_path', type=str, 
                       default='../../models/Qwen3-4B-Instruct-2507',
                       help='Path to LLM model')
    parser.add_argument('--llm_name', type=str, default='qwen3-4b',
                       help='Model name for API')
    parser.add_argument('--embedding_model_path', type=str,
                       default='../../models/bge-m3',
                       help='Path to embedding model')
    parser.add_argument('--router_model_path', type=str,
                       default='../../HybridRAG/H-STAR/router/wikitq',
                       help='Path to router model')
    parser.add_argument('--check_model_path', type=str,
                       default='../../HybridRAG/H-STAR/check/wikitq',
                       help='Path to check model')
    
    # Dataset parameters
    parser.add_argument('--tablebench_jsonl_path', type=str,
                       default='../datasets/TableBench/tablebench_math_long_98.json',
                       help='Path to TableBench JSON file (long context version)')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/context_length_exp',
                       help='Output directory')
    
    # Experiment parameters
    parser.add_argument('--target_lengths', type=str, default='8000,16000,32000,64000,128000',
                       help='Comma-separated target token lengths')
    parser.add_argument('--first_n', type=int, default=-1,
                       help='Only process first N samples (-1 for all)')
    
    # Pipeline parameters
    parser.add_argument('--tau', type=float, default=0.82, help='Router threshold')
    parser.add_argument('--check_tau', type=float, default=0.8, help='Check model threshold')
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    
    # LLM parameters
    parser.add_argument('--select_sample_num', type=int, default=2)
    parser.add_argument('--sql_sample_num', type=int, default=3)
    parser.add_argument('--llm_concurrency', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    
    # API parameters
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--api_base', type=str, default='http://localhost:8000/v1')
    parser.add_argument('--api_key', type=str, default='api-key-qwen3')
    
    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================

def load_tablebench_dataset(jsonl_path: str, first_n: int = -1) -> List[Dict]:
    """Load TableBench dataset from JSON or JSONL file."""
    print(f"Loading dataset from {jsonl_path}...")
    
    if jsonl_path.endswith('.json'):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    if first_n > 0:
        data = data[:first_n]
    
    print(f"Loaded {len(data)} samples")
    return data


def make_unique_columns(columns: List[str]) -> List[str]:
    """Make column names unique by adding suffixes to duplicates."""
    seen = {}
    result = []
    for col in columns:
        col_str = str(col).strip() if col else "unnamed"
        if not col_str:
            col_str = "unnamed"
        col_str = col_str.replace('\n', ' ').replace('\r', ' ')
        col_str = re.sub(r'\s+', ' ', col_str).strip()
        if col_str in seen:
            seen[col_str] += 1
            result.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            result.append(col_str)
    return result


def tablebench_table_to_df(item: Dict) -> pd.DataFrame:
    """Convert TableBench table format to pandas DataFrame."""
    columns = item['table']['columns']
    data = item['table']['data']
    unique_columns = make_unique_columns(columns)
    return pd.DataFrame(data, columns=unique_columns)


def table_to_str_tablebench(df: pd.DataFrame) -> str:
    """Format DataFrame as a tab-separated string."""
    temp_df = df.copy()
    if 'row_id' not in temp_df.columns:
        temp_df.insert(0, 'row_id', temp_df.index)
    return temp_df.to_string(index=False)


def truncate_table_to_tokens(df: pd.DataFrame, max_tokens: int, tokenizer) -> pd.DataFrame:
    """
    Dynamically truncate table to fit within max_tokens.
    
    Strategy:
    1. First try to reduce columns (keep most important ones)
    2. Then reduce rows if still over limit
    """
    if max_tokens <= 0:
        return df
    
    current_df = df.copy()
    
    # Estimate base overhead (schema, question, etc.) ~500 tokens
    effective_max = max_tokens - 500
    
    while True:
        table_str = table_to_str_tablebench(current_df)
        tokens = len(tokenizer.encode(table_str))
        
        if tokens <= effective_max:
            break
        
        # Priority: reduce columns first, then rows
        if len(current_df.columns) > 3:
            # Remove last column
            current_df = current_df.iloc[:, :-1]
        elif len(current_df) > 3:
            # Remove last row
            current_df = current_df.iloc[:-1, :]
        else:
            # Can't reduce further
            break
    
    return current_df


def build_prompt_from_df(tablebench_data: List[Dict], table_df: pd.DataFrame, 
                         index: int, template_path: str) -> str:
    """Build prompt for TableBench item."""
    item = tablebench_data[index]
    
    template_full_path = os.path.join(os.path.dirname(__file__), template_path)
    if os.path.exists(template_full_path):
        with open(template_full_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = ""
    
    table_title = item.get('id', f'Table_{index}')
    question = item.get('question', '')
    
    # Format table as string
    table_str = table_to_str_tablebench(table_df)
    
    # Build CREATE TABLE schema
    columns = table_df.columns.tolist()
    schema_cols = []
    if 'row_id' not in columns:
        schema_cols.append("row_id int")
    
    for col in columns:
        schema_cols.append(f"`{col}` text")
    
    col_defs = ",\n\t".join(schema_cols)
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(table_title))
    schema = f"CREATE TABLE {table_name}(\n\t{col_defs})"
    
    all_cols = ['row_id'] + columns if 'row_id' not in columns else columns
    
    input_section = f"""
<input>
{schema}
/*
SELECT * FROM w;
{table_str}
*/
columns: {all_cols}
Q: {question}
<output>"""
    
    if template:
        prompt = template.strip() + "\n" + input_section
    else:
        prompt = f"Table ID: {table_title}\n\n{table_str}\n\nQuestion: {question}"
    
    return prompt


def run_single_config(args, tablebench_data: List[Dict], full_tables: Dict[int, pd.DataFrame],
                     truncated_tables: Dict[int, pd.DataFrame], max_tokens: int,
                     config_output_path: str) -> Dict:
    """
    Run the full pipeline for a single token configuration.
    
    Key: SQL execution uses full_tables, LLM prompts use truncated_tables
    """
    os.makedirs(config_output_path, exist_ok=True)
    
    results = {
        'max_tokens': max_tokens,
        'num_samples': len(tablebench_data),
        'predictions': [],
        'golds': [],
    }
    
    ALL_LABELS = ['Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5']
    
    # ========================================================================
    # Step 1: Build Router Query (using TRUNCATED tables)
    # ========================================================================
    print(f"\n  [Config {max_tokens}] Building Router Query...")
    semantic_router = {}
    for idx in range(len(tablebench_data)):
        item = tablebench_data[idx]
        semantic_router[idx] = {
            'query': item.get('question', ''),
            'title': item.get('id', f'Table_{idx}'),
            'table': truncated_tables[idx],  # TRUNCATED for router
            'label': []
        }
    
    router_query_file = f'{config_output_path}/router_query.pkl'
    with open(router_query_file, 'wb') as f:
        pickle.dump(semantic_router, f)
    
    # ========================================================================
    # Step 2: Construct Database (using FULL tables for SQL)
    # ========================================================================
    print(f"  [Config {max_tokens}] Constructing Database...")
    table_titles = [tablebench_data[i].get('id', f'Table_{i}') for i in range(len(tablebench_data))]
    tables_for_db = [full_tables[i] for i in range(len(tablebench_data))]  # FULL tables
    
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    
    # ========================================================================
    # Step 3: Router Inference
    # ========================================================================
    print(f"  [Config {max_tokens}] Router Inference...")
    router_result_file = f'{config_output_path}/inference_result.pkl'
    
    cmd = f"python inference_router.py --input_path {router_query_file} " \
          f"--model_path {args.router_model_path} " \
          f"--output_path {router_result_file}"
    os.system(cmd)
    
    if os.path.exists(router_result_file):
        with open(router_result_file, 'rb') as f:
            error_analysis_row = pickle.load(f)
    else:
        # Default routing
        error_analysis_row = {i: {'Base': 0.5, 'Select_Row': 0.5, 'Select_Column': 0.5, 
                                  'Execute_SQL': 0.5, 'RAG_20_5': 0.5} 
                             for i in range(len(tablebench_data))}
    
    ranked_result = process_error_analysis_list(error_analysis_row, truncate=True, tau=args.tau)
    
    # ========================================================================
    # Step 4: Build LLM Queries (using TRUNCATED tables)
    # ========================================================================
    print(f"  [Config {max_tokens}] Building LLM Queries...")
    LLM_query_list = {method: {'index': [], 'query': []} for method in ALL_LABELS}
    
    for idx in range(len(tablebench_data)):
        for method in ALL_LABELS:
            if method in ranked_result[idx]:
                LLM_query_list[method]['index'].append(idx)
                
                if method == 'Select_Column':
                    prompt = build_prompt_from_df(
                        tablebench_data, truncated_tables[idx], idx,  # TRUNCATED
                        template_path='../prompts/col_select_sql.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Select_Row':
                    prompt = build_prompt_from_df(
                        tablebench_data, truncated_tables[idx], idx,  # TRUNCATED
                        template_path='../prompts/row_select_sql.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Execute_SQL':
                    prompt = build_prompt_from_df(
                        tablebench_data, truncated_tables[idx], idx,  # TRUNCATED
                        template_path='../prompts/sql_reason_wtq.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
    
    # ========================================================================
    # Step 5: LLM Inference
    # ========================================================================
    print(f"  [Config {max_tokens}] LLM Inference...")
    
    llm_results = {}
    for method in ['Select_Row', 'Select_Column', 'Execute_SQL']:
        if len(LLM_query_list[method]['query']) > 0:
            responses, _, _ = infer_prompts(
                LLM_query_list[method]['query'],
                sample_num=args.sql_sample_num if method == 'Execute_SQL' else args.select_sample_num,
                temperature=args.temperature,
                top_p=args.top_p,
                llm_name=args.llm_name,
                api_base=args.api_base,
                api_key=args.api_key,
                concurrency=args.llm_concurrency
            )
            llm_results[method] = {
                LLM_query_list[method]['index'][i]: responses[i]
                for i in range(len(responses))
            }
        else:
            llm_results[method] = {}
    
    # ========================================================================
    # Step 6: Execute SQL (using FULL tables)
    # ========================================================================
    print(f"  [Config {max_tokens}] Executing SQL...")
    sql_results = {}
    
    for idx in llm_results.get('Execute_SQL', {}):
        responses = llm_results['Execute_SQL'][idx]
        full_df = full_tables[idx]  # FULL table for execution
        
        # Try each SQL response
        for resp in responses:
            sql_match = re.search(r'SQL:\s*(.+?)(?:\n|$)', resp, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
                try:
                    result = executor.execute_sql(sql_query, db)
                    if result is not None and len(result) > 0:
                        sql_results[idx] = result
                        break
                except Exception:
                    continue
    
    # ========================================================================
    # Step 7: Final QA (using TRUNCATED tables + SQL context)
    # ========================================================================
    print(f"  [Config {max_tokens}] Final QA...")
    
    final_prompts = []
    for idx in range(len(tablebench_data)):
        item = tablebench_data[idx]
        truncated_df = truncated_tables[idx]
        
        # Build evidence from SQL result if available
        evidence = ""
        if idx in sql_results:
            evidence = f"\nSQL Result:\n{sql_results[idx].to_string(index=False)}\n"
        
        prompt = build_prompt_from_df(
            tablebench_data, truncated_df, idx,  # TRUNCATED
            template_path='../prompts/text_reason_wtq.txt'
        )
        if evidence:
            prompt = prompt.replace('<output>', f'{evidence}<output>')
        
        final_prompts.append(prompt)
    
    final_responses, _, _ = infer_prompts(
        final_prompts,
        sample_num=1,
        temperature=0,
        top_p=1,
        llm_name=args.llm_name,
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.llm_concurrency
    )
    
    # ========================================================================
    # Step 8: Evaluate
    # ========================================================================
    print(f"  [Config {max_tokens}] Evaluating...")
    
    preds = []
    golds = []
    
    length_result = {}
    for idx, item in enumerate(tablebench_data):
        pred = final_responses[idx][0] if isinstance(final_responses[idx], list) else final_responses[idx]
        gold = item.get('answer', '')
        table_id = item.get('table_id', '')
        if table_id not in length_result:
            length_result[table_id] = {
                "preds": [],
                "golds": []
            }
        length_result[table_id]['preds'].append(str(pred))
        length_result[table_id]['golds'].append(str(gold))
        preds.append(str(pred))
        golds.append(str(gold))
    
    eval_results = evaluate_tablebench_predictions(preds, golds)
    
    with open(f"{config_output_path}/preds_and_golds.json", 'w') as f_result:
        json.dump(length_result, f_result, indent=2)

    for table_id, table_info in length_result.items():
        table_preds = table_info['preds']
        table_golds = table_info['golds']
        res = evaluate_tablebench_predictions(table_preds, table_golds)
        print(f"{table_id}, result: {res}")

    results['predictions'] = preds
    results['golds'] = golds
    results['rouge_l'] = eval_results['avg_rouge_l']
    results['accuracy'] = eval_results.get('accuracy_at_0.5', 0)
    
    # Save config results
    with open(f'{config_output_path}/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Context Length Experiment for TableBench")
    print("=" * 60)
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    # Parse target lengths
    target_lengths = [int(x.strip()) for x in args.target_lengths.split(',')]
    print(f"Target lengths: {target_lengths}")
    
    # Load tokenizer for length estimation
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    
    # Load data
    tablebench_data = load_tablebench_dataset(args.tablebench_jsonl_path, args.first_n)
    
    # Preprocess FULL tables
    print("\nPreprocessing tables...")
    full_tables = {}
    for idx, item in enumerate(tqdm(tablebench_data, desc="Processing")):
        full_tables[idx] = tablebench_table_to_df(item)
    
    # Run experiment for each token configuration
    all_results = {}
    runtimes = {}
    for max_tokens in target_lengths:
        print(f"\n{'='*60}")
        print(f"Running configuration: {max_tokens} tokens")
        print(f"{'='*60}")
        
        _t0 = time.perf_counter()

        # Truncate tables for this configuration
        truncated_tables = {}
        for idx in range(len(tablebench_data)):
            truncated_tables[idx] = truncate_table_to_tokens(
                full_tables[idx], max_tokens, tokenizer
            )
        
        # Compute average truncated size
        avg_cols = np.mean([len(truncated_tables[i].columns) for i in range(len(tablebench_data))])
        avg_rows = np.mean([len(truncated_tables[i]) for i in range(len(tablebench_data))])
        print(f"Truncated table stats: avg {avg_cols:.1f} cols, {avg_rows:.1f} rows")
        
        config_output_path = f'{args.tmp_save_path}/config_{max_tokens}'
        
        results = run_single_config(
            args, tablebench_data, full_tables, truncated_tables,
            max_tokens, config_output_path
        )
        
        _t1 = time.perf_counter()
        all_results[max_tokens] = results
        runtimes[max_tokens] = (_t1 - _t0)
        print(f"\n  Runtime: {runtimes[max_tokens]:.6f}")
        print(f"\n  ROUGE-L @ {max_tokens} tokens: {results['rouge_l']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Token Length':<15} {'ROUGE-L':<10} {'Accuracy':<10} {'Time':<10}")
    print("-" * 35)
    for max_tokens in target_lengths:
        r = all_results[max_tokens]
        print(f"{max_tokens:<15} {r['rouge_l']:.4f}     {r['accuracy']:.4f}     {runtimes[max_tokens]:.6f}")
    
    # Save summary
    summary = {
        'target_lengths': target_lengths,
        'results': {k: {'rouge_l': v['rouge_l'], 'accuracy': v['accuracy']} 
                   for k, v in all_results.items()}
    }
    with open(f'{args.tmp_save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.tmp_save_path}/")


if __name__ == "__main__":
    main()
