#!/usr/bin/env python3
"""
Router Model Benchmark Pipeline for TableBench
Iterates over different router models to benchmark their performance.

Based on run_full_pipeline_tablebench.py structure.
Keeps Check Model and LLM constant, only varies Router Model.
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
from typing import Dict, Any, List
import re
from glob import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_llm import infer_prompts
from utils.schedule_utils import (
    table_to_str_sql, find_intersection, sql_data_cleaning, find_intersection_and_add_row_id,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Router Model Benchmark Pipeline")
    
    # Model paths
    parser.add_argument('--llm_path', type=str, 
                       default='../../models/Qwen3-4B-Instruct-2507',
                       help='Path to LLM model (constant)')
    parser.add_argument('--llm_name', type=str, default='qwen3-4b',
                       help='Model name for API')
    parser.add_argument('--embedding_model_path', type=str,
                       default='../../models/bge-m3',
                       help='Path to embedding model')
    parser.add_argument('--router_models_dir', type=str,
                       default='../models',
                       help='Directory containing router models to benchmark')
    parser.add_argument('--router_model_pattern', type=str,
                       default='*router*',
                       help='Glob pattern to filter router models')
    parser.add_argument('--check_model_path', type=str,
                       default='../../HybridRAG/H-STAR/check/wikitq',
                       help='Path to check model (constant)')
    
    # Dataset parameters
    parser.add_argument('--tablebench_jsonl_path', type=str,
                       default='../datasets/TableBench/TableBench.jsonl',
                       help='Path to TableBench JSONL file')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/router_benchmark',
                       help='Output directory')
    
    # Pipeline parameters
    parser.add_argument('--tau', type=float, default=0.82, help='Router threshold')
    parser.add_argument('--check_tau', type=float, default=0.8, help='Check model threshold')
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--first_n', type=int, default=-1, help='Only process first N samples')
    
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
    
    # Skip options
    parser.add_argument('--skip_preprocess', action='store_true')
    
    return parser.parse_args()


# ============================================================================
# Helper Functions (same as run_full_pipeline_tablebench.py)
# ============================================================================

def load_tablebench_dataset(jsonl_path: str, first_n: int = -1) -> List[Dict]:
    """Load TableBench dataset from JSONL file."""
    print(f"Loading dataset from {jsonl_path}...")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if first_n > 0 and len(data) >= first_n:
                    break
    print(f"Loaded {len(data)} samples")
    return data


def make_unique_columns(columns: List[str]) -> List[str]:
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
    columns = item['table']['columns']
    data = item['table']['data']
    unique_columns = make_unique_columns(columns)
    return pd.DataFrame(data, columns=unique_columns)


def table_to_str_tablebench(df: pd.DataFrame) -> str:
    temp_df = df.copy()
    if 'row_id' not in temp_df.columns:
        temp_df.insert(0, 'row_id', temp_df.index)
    return temp_df.to_string(index=False)


def build_tablebench_prompt_from_df(tablebench_data: List[Dict], table_df: pd.DataFrame, 
                                    index: int, template_path: str) -> str:
    item = tablebench_data[index]
    
    template_full_path = os.path.join(os.path.dirname(__file__), template_path)
    if os.path.exists(template_full_path):
        with open(template_full_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = ""
    
    table_title = item.get('id', f'Table_{index}')
    question = item.get('question', '')
    table_str = table_to_str_tablebench(table_df)
    
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


def discover_router_models(base_dir: str, pattern: str) -> List[str]:
    """Discover router models in the given directory."""
    search_path = os.path.join(base_dir, pattern)
    candidates = glob(search_path)
    
    # Filter to only directories that look like model directories
    models = []
    for path in candidates:
        if os.path.isdir(path):
            # Check if it contains model files
            if any(f.endswith(('.bin', '.safetensors', 'config.json')) 
                   for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
                models.append(path)
    
    return sorted(models)


def run_single_router(args, tablebench_data: List[Dict], 
                     tablebench_df_processed: Dict[int, pd.DataFrame],
                     db: NeuralDB, executor: Executor,
                     router_model_path: str, output_path: str) -> Dict:
    """
    Run the full pipeline with a single router model.
    Returns evaluation metrics.
    """
    os.makedirs(output_path, exist_ok=True)
    
    router_name = os.path.basename(router_model_path)
    print(f"\n{'='*60}")
    print(f"Testing Router: {router_name}")
    print(f"{'='*60}")
    
    ALL_LABELS = ['Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5']
    
    # Step 1: Build Router Query
    print("  [1] Building Router Query...")
    semantic_router = {}
    for idx in range(len(tablebench_data)):
        item = tablebench_data[idx]
        semantic_router[idx] = {
            'query': item.get('question', ''),
            'title': item.get('id', f'Table_{idx}'),
            'table': tablebench_df_processed[idx],
            'label': []
        }
    
    router_query_file = f'{output_path}/router_query.pkl'
    with open(router_query_file, 'wb') as f:
        pickle.dump(semantic_router, f)
    
    # Step 2: Router Inference
    print("  [2] Router Inference...")
    router_result_file = f'{output_path}/inference_result.pkl'
    
    cmd = f"python inference_router.py --input_path {router_query_file} " \
          f"--model_path {router_model_path} " \
          f"--output_path {router_result_file}"
    os.system(cmd)
    
    if os.path.exists(router_result_file):
        with open(router_result_file, 'rb') as f:
            error_analysis_row = pickle.load(f)
    else:
        print("  Warning: Router result not found, using default routing")
        error_analysis_row = {i: {'Base': 0.5, 'Select_Row': 0.5, 'Select_Column': 0.5, 
                                  'Execute_SQL': 0.5, 'RAG_20_5': 0.5} 
                             for i in range(len(tablebench_data))}
    
    ranked_result = process_error_analysis_list(error_analysis_row, truncate=True, tau=args.tau)
    
    # Step 3: Build LLM Queries
    print("  [3] Building LLM Queries...")
    LLM_query_list = {method: {'index': [], 'query': []} for method in ALL_LABELS}
    
    for idx in range(len(tablebench_data)):
        for method in ALL_LABELS:
            if method in ranked_result[idx]:
                LLM_query_list[method]['index'].append(idx)
                
                if method == 'Select_Column':
                    prompt = build_tablebench_prompt_from_df(
                        tablebench_data, tablebench_df_processed[idx], idx,
                        template_path='../prompts/col_select_sql.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Select_Row':
                    prompt = build_tablebench_prompt_from_df(
                        tablebench_data, tablebench_df_processed[idx], idx,
                        template_path='../prompts/row_select_sql.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Execute_SQL':
                    prompt = build_tablebench_prompt_from_df(
                        tablebench_data, tablebench_df_processed[idx], idx,
                        template_path='../prompts/sql_reason_wtq.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
    
    # Step 4: LLM Inference
    print("  [4] LLM Inference...")
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
    
    # Step 5: Execute SQL
    print("  [5] Executing SQL...")
    processed_table = {'Base': tablebench_df_processed}
    
    # Process SQL results
    Execute_SQL_count = []
    for idx in llm_results.get('Execute_SQL', {}):
        responses = llm_results['Execute_SQL'][idx]
        df = tablebench_df_processed[idx]
        
        for resp in responses:
            sql_match = re.search(r'SQL:\s*(.+?)(?:\n|$)', resp, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
                try:
                    result = executor.execute_sql(sql_query, db)
                    if result is not None and len(result) > 0:
                        Execute_SQL_count.append({
                            'id': idx,
                            'sql': f"SQL: {sql_query}\n",
                            'table': result
                        })
                        break
                except Exception:
                    continue
    
    processed_table['Execute_SQL_count'] = Execute_SQL_count
    
    # Step 6: Final QA
    print("  [6] Final QA...")
    final_prompts = []
    for idx in range(len(tablebench_data)):
        prompt = build_tablebench_prompt_from_df(
            tablebench_data, tablebench_df_processed[idx], idx,
            template_path='../prompts/text_reason_wtq.txt'
        )
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
    
    # Step 7: Evaluate
    print("  [7] Evaluating...")
    preds = []
    golds = []
    
    for idx, item in enumerate(tablebench_data):
        pred = final_responses[idx][0] if isinstance(final_responses[idx], list) else final_responses[idx]
        gold = item.get('answer', '')
        preds.append(str(pred))
        golds.append(str(gold))
    
    eval_results = evaluate_tablebench_predictions(preds, golds)
    
    results = {
        'router_model': router_name,
        'router_path': router_model_path,
        'num_samples': len(tablebench_data),
        'rouge_l': eval_results['avg_rouge_l'],
        'accuracy': eval_results.get('accuracy_at_0.5', 0),
        'routing_distribution': pd.DataFrame([str(r) for r in ranked_result.values()]).value_counts().to_dict()
    }
    
    # Save results
    with open(f'{output_path}/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ROUGE-L: {results['rouge_l']:.4f}, Accuracy: {results['accuracy']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Router Model Benchmark Pipeline")
    print("=" * 60)
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    # Discover router models
    router_models = discover_router_models(args.router_models_dir, args.router_model_pattern)
    
    if not router_models:
        # If no models found with pattern, list all directories
        print(f"No models found with pattern '{args.router_model_pattern}' in {args.router_models_dir}")
        print("Listing all directories...")
        router_models = [d for d in glob(os.path.join(args.router_models_dir, '*')) if os.path.isdir(d)]
    
    print(f"Found {len(router_models)} router models to test:")
    for m in router_models:
        print(f"  - {os.path.basename(m)}")
    
    if not router_models:
        print("ERROR: No router models found!")
        return
    
    # Load data
    tablebench_data = load_tablebench_dataset(args.tablebench_jsonl_path, args.first_n)
    
    # Preprocess tables
    preprocess_file = f'{args.tmp_save_path}/tablebench_df_processed.npy'
    
    if not args.skip_preprocess or not os.path.exists(preprocess_file):
        print("\nPreprocessing tables...")
        tablebench_df_processed = {}
        for idx, item in enumerate(tqdm(tablebench_data, desc="Processing")):
            tablebench_df_processed[idx] = tablebench_table_to_df(item)
        np.save(preprocess_file, tablebench_df_processed)
    else:
        tablebench_df_processed = np.load(preprocess_file, allow_pickle=True).item()
    
    # Construct database
    print("\nConstructing database...")
    table_titles = [tablebench_data[i].get('id', f'Table_{i}') for i in range(len(tablebench_data))]
    tables_for_db = [tablebench_df_processed[i] for i in range(len(tablebench_data))]
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    
    # Run benchmark for each router model
    all_results = {}
    
    for router_model_path in router_models:
        router_name = os.path.basename(router_model_path)
        output_path = f'{args.tmp_save_path}/{router_name}'
        
        try:
            results = run_single_router(
                args, tablebench_data, tablebench_df_processed,
                db, executor, router_model_path, output_path
            )
            all_results[router_name] = results
        except Exception as e:
            print(f"ERROR testing {router_name}: {e}")
            all_results[router_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Router Model':<40} {'ROUGE-L':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for router_name, results in sorted(all_results.items(), key=lambda x: x[1].get('rouge_l', 0), reverse=True):
        if 'error' in results:
            print(f"{router_name:<40} {'ERROR':<10} {results['error'][:20]}")
        else:
            print(f"{router_name:<40} {results['rouge_l']:.4f}     {results['accuracy']:.4f}")
    
    # Save summary
    with open(f'{args.tmp_save_path}/summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.tmp_save_path}/")


if __name__ == "__main__":
    main()
