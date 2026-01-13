#!/usr/bin/env python3
"""
Complete H-STAR Pipeline for NIAT Dataset (Nested/Hierarchical Tables)
Fully aligned with run_full_pipeline_wikitq.py

Pipeline Steps:
1. Data Preprocessing (NIAT-specific: flatten nested tables)
2. Build Router Query File
3. Construct Database
4. Router Model Inference
5. Parse Router Results & Organize LLM Query List
6. Execute RAG Task
7. Execute LLM Queries (Select_Row, Select_Column)
8. Execute SQL Queries
9. SQL Parse and Execute
10. Check Model Iteration
11. Add Missing Execute_SQL
12. Generate Final QA Prompts
13. Execute Final QA and Evaluate
"""

import os
import sys
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List
from datetime import datetime
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports - same as wikitq
from utils.async_llm import infer_prompts
from utils.schedule_utils import (
    table_to_str, table_to_str_sql,
    find_intersection_and_add_row_id, Prepare_Data_for_Operator_Sequence,
    format_document, batch_rerank_scores, ROLLBACK,
    merge_clean_and_format_df_dict, retrieve_rows_by_subtables,
    process_error_analysis_list
)
from utils.evaluator import Evaluator, niat_match_func_for_samples
from utils.prompt_generate import (
    format_table_prompt, fix_sql_query, filter_dataframe_from_responses,
    match_subtables, retrieve_rows_by_subtables
)
from utils.multi_db_v2 import NeuralDB, Executor
from FlagEmbedding import FlagReranker
import multiprocessing as mp

# Multiprocessing setup - same as wikitq
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="H-STAR Full Pipeline for NIAT")
    
    # Model paths - same as wikitq
    parser.add_argument('--llm_path', type=str, 
                       default='/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/',
                       help='Path to LLM model')
    parser.add_argument('--embedding_model_path', type=str,
                       default='/data/workspace/yanmy/models/bge-m3',
                       help='Path to embedding model')
    parser.add_argument('--router_model_path', type=str,
                       default='/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/',
                       help='Path to router model')
    parser.add_argument('--check_model_path', type=str,
                       default='/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/',
                       help='Path to check model')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='niat', help='Dataset name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/niat',
                       help='Temporary save path for intermediate results')
    parser.add_argument('--niat_json_path', type=str,
                       default='datasets/NIAT/sampled_qa_pairs_4000_fixed.json',
                       help='Path to NIAT JSON file')
    
    # Pipeline parameters - same as wikitq
    parser.add_argument('--tau', type=float, default=0.82, help='Router threshold')
    parser.add_argument('--check_tau', type=float, default=0.8, help='Check model threshold')
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    
    # vLLM parameters - same as wikitq
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--max_model_len', type=int, default=23000,
                       help='Maximum model length')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                       help='GPU memory utilization')
    parser.add_argument('--max_num_seqs', type=int, default=256,
                       help='Maximum number of sequences')
    
    # Sampling parameters - same as wikitq
    parser.add_argument('--select_sample_num', type=int, default=2,
                       help='Number of samples for Select_Row/Select_Column')
    parser.add_argument('--sql_sample_num', type=int, default=3,
                       help='Number of samples for Execute_SQL')
    parser.add_argument('--llm_concurrency', type=int, default=32,
                       help='Max concurrent requests to vLLM API')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8,
                       help='Sampling top_p')
    
    # Execution control - same as wikitq
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--skip_router', action='store_true',
                       help='Skip router inference if already done')
    parser.add_argument('--skip_rag', action='store_true',
                       help='Skip RAG if already done')
    parser.add_argument('--first_n', type=int, default=-1,
                       help='Only process first N samples (-1 for all)')
    parser.add_argument('--use_api', action='store_true',
                       help='Use async API instead of local vLLM')
    
    return parser.parse_args()


# ============================================================================
# NIAT-Specific Functions
# ============================================================================

def load_niat_dataset(json_path: str, first_n: int = -1) -> List[Dict]:
    """Load NIAT dataset from JSON file."""
    print(f"Loading NIAT dataset from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
        col_str = col_str.lower().replace('\n', ' ').replace('\r', ' ')
        
        if col_str in seen:
            seen[col_str] += 1
            result.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            result.append(col_str)
    
    return result


def flatten_hierarchical_table(table_rows: List[List[str]], 
                                table_structure: str = "vertical") -> pd.DataFrame:
    """Flatten hierarchical/nested table by forward-filling empty cells."""
    if not table_rows or len(table_rows) < 2:
        if len(table_rows) == 1:
            header = make_unique_columns(table_rows[0])
            return pd.DataFrame(columns=header)
        return pd.DataFrame()
    
    header = make_unique_columns(table_rows[0])
    num_cols = len(header)
    
    # Normalize data rows to match header column count
    normalized_rows = []
    for row in table_rows[1:]:
        if len(row) < num_cols:
            row = list(row) + [''] * (num_cols - len(row))
        elif len(row) > num_cols:
            row = row[:num_cols]
        normalized_rows.append(row)
    
    df = pd.DataFrame(normalized_rows, columns=header)
    
    # For hierarchical tables, forward-fill left-most grouping columns
    if table_structure == "hierarchical":
        for col_idx in range(min(2, len(df.columns))):
            col_name = df.columns[col_idx]
            col_values = df[col_name].values
            
            prev_non_empty = False
            empty_after_non_empty = 0
            for val in col_values:
                val_str = str(val) if not isinstance(val, (list, np.ndarray)) else ""
                val_str = val_str.strip() if isinstance(val_str, str) else ""
                is_empty = (val_str == "" or val_str == " ")
                if is_empty and prev_non_empty:
                    empty_after_non_empty += 1
                if not is_empty:
                    prev_non_empty = True
            
            if empty_after_non_empty > len(col_values) * 0.2:
                df[col_name] = df[col_name].replace(r'^\s*$', np.nan, regex=True)
                df[col_name] = df[col_name].ffill()
                df[col_name] = df[col_name].fillna('')
    
    return df



def table_to_str_niat(df: pd.DataFrame) -> str:
    """
    Format DataFrame as a tab-separated string, matching NIAT prompt demonstrations.
    Includes row_id (index) as the first column.
    """
    # Create a copy to avoid modifying original
    temp_df = df.copy()
    
    # Insert row_id as first column if not present
    if 'row_id' not in temp_df.columns:
        temp_df.insert(0, 'row_id', temp_df.index)
        
    # Convert to tab-separated string
    # align columns for readability if possible, or just simple tab sep
    # The prompt examples look like they use tabs or fixed spacing. 
    # to_markdown(index=False, tablefmt="plain") might be close but to_string is safer
    
    # Using to_string with header and index=False (since we added row_id column)
    # This aligns columns nicely which matches 'tab-separated' look in prompts
    return temp_df.to_string(index=False)


def build_niat_prompt_from_df(niat_data: List[Dict], table_df: pd.DataFrame, 
                               index: int, template_path: str = '../prompts/col_select_niat.txt',
                               processed: bool = True) -> str:
    """
    Build prompt for NIAT item.
    
    Template structure:
    - Instructions + demonstrations (from template file)
    - Current input: <input> + table schema + table data + question
    
    The template contains demonstrations, and we APPEND the current query's
    table and question at the end.
    """
    item = niat_data[index]
    
    template_full_path = os.path.join(os.path.dirname(__file__), template_path)
    if os.path.exists(template_full_path):
        with open(template_full_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = ""
    
    table_title = item.get('table_title', 'Table') or item.get('table_id', 'Table')
    question = item.get('question', '')
    
    # Format table as string using NIAT-specific format
    table_str = table_to_str_niat(table_df)
    
    # Build CREATE TABLE schema from DataFrame columns
    # Note: row_id is usually implicit in schema in examples or explicit.
    # Examples have "row_id int" in schema.
    columns = table_df.columns.tolist()
    
    # If row_id is not in columns, we add it to schema definition as in examples
    schema_cols = []
    if 'row_id' not in columns:
        schema_cols.append("row_id int")
        
    for col in columns:
        # Simple type inference or default to text
        # Examples use 'text' or 'int'. We'll stick to text for robustness
        # unless we want to be fancy. 'text' is safe.
        schema_cols.append(f"`{col}` text")
        
    col_defs = ",\n\t".join(schema_cols)
    schema = f"CREATE TABLE {table_title.replace(' ', '_').replace('-', '_')}(\n\t{col_defs})"
    
    # List of columns for "columns: [...]" line
    # Examples include row_id in this list
    all_cols = ['row_id'] + columns if 'row_id' not in columns else columns
    
    # Build the input section with current query data
    # Format like the demonstrations: schema + table data + columns list + question
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
    
    # Append input section to template (after demonstrations)
    if template:
        prompt = template.strip() + "\n" + input_section
    else:
        # Fallback if no template
        prompt = f"Table Title: {table_title}\n\n{table_str}\n\nQuestion: {question}"
    
    return prompt


def Prepare_Data_for_Operator_Sequence_NIAT(index: int, sequence: List[str], 
                                            niat_data: List[Dict], processed_table: Dict) -> Dict:
    """
    NIAT-specific version of Prepare_Data_for_Operator_Sequence.
    Handles NIAT data structure which differs from WikiTQ HuggingFace dataset.
    """
    item = niat_data[index]
    
    data_entry = {
        'id': item.get('id', str(index)),
        'query': item.get('question', ''),
        'title': item.get('table_title', f'Table_{index}') or f'Table_{index}',
    }
    
    # Handle Execute_SQL
    if 'Execute_SQL' in sequence:
        sql_list = [sql_count for sql_count in processed_table.get('Execute_SQL_count', []) 
                   if sql_count['id'] == index]
        if len(sql_list) == 0:
            SQL = ''
        else:
            sql_command = sql_list[0]
            # Use NIAT-specific table formatting
            SQL = sql_command['sql'] + table_to_str_niat(sql_command['table'])
        data_entry['SQL'] = SQL
    else:
        data_entry['SQL'] = ''
    
    # Get table based on extraction sequence
    extraction_sequence = [s for s in sequence if s != 'Execute_SQL']
    
    if len(extraction_sequence) == 1:
        method = extraction_sequence[0]
        if method in processed_table and index in processed_table[method]:
            data_entry['table'] = processed_table[method][index]
        else:
            data_entry['table'] = processed_table['Base'].get(index, pd.DataFrame())
    elif len(extraction_sequence) == 2:
        table_1 = processed_table.get(extraction_sequence[0], {}).get(index, pd.DataFrame())
        table_2 = processed_table.get(extraction_sequence[1], {}).get(index, pd.DataFrame())
        if len(table_1) > 0 and len(table_2) > 0:
            table_intersection = find_intersection_and_add_row_id(table_1, table_2)
            data_entry['table'] = table_intersection
        else:
            data_entry['table'] = table_1 if len(table_1) > 0 else table_2
    else:
        data_entry['table'] = processed_table['Base'].get(index, pd.DataFrame())
    
    return data_entry

# ============================================================================
# Main Pipeline - Aligned with WikiTQ
# ============================================================================

def main():
    args = parse_args()
    
    print("=" * 80)
    print("H-STAR Full Pipeline for NIAT")
    print("=" * 80)
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    ALL_LABELS = ['Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5']
    
    timeline = {}
    overall_start = time.perf_counter()
    metrics_rows = {}
    summaries = {}
    
    # ========================================================================
    # Step 1: Data Preprocessing (NIAT-specific)
    # KEY FIX: Use table_id as key, not sequential index
    # ========================================================================
    print("\n[Step 1] Data Preprocessing...")
    _t1 = time.perf_counter()
    
    preprocess_file = f'{args.tmp_save_path}/niat_df_processed.npy'
    
    # Load NIAT data (list of QA samples)
    niat_data = load_niat_dataset(args.niat_json_path, args.first_n)
    
    if not args.skip_preprocess or not os.path.exists(preprocess_file):
        # Process tables with flattening - KEY: use table_id as key
        print("Processing nested tables with flattening (table_id indexed)...")
        niat_df_processed = {}  # {table_id: DataFrame}
        
        # Only process each unique table once
        for item in tqdm(niat_data, desc="Flattening tables"):
            table_id = item['table_id']
            if table_id not in niat_df_processed:
                table_rows = item.get('table_rows', [])
                table_structure = item.get('table_structure', 'vertical')
                df = flatten_hierarchical_table(table_rows, table_structure)
                niat_df_processed[table_id] = df
        
        print(f"Processed {len(niat_df_processed)} unique tables from {len(niat_data)} QA samples")
        np.save(preprocess_file, niat_df_processed)
        print(f"Saved preprocessed data to {preprocess_file}")
    else:
        niat_df_processed = np.load(preprocess_file, allow_pickle=True).item()
        print(f"Loaded preprocessed data from {preprocess_file}")
    
    print(f"QA samples: {len(niat_data)}, Unique tables: {len(niat_df_processed)}")
    
    timeline['Step 1 - Data Preprocessing'] = time.perf_counter() - _t1
    print(f"  [Timing] Step 1: {timeline['Step 1 - Data Preprocessing']:.2f}s")
    
    # ========================================================================
    # Step 2: Build Router Query File
    # KEY FIX: Use qa_idx for QA samples, table_id for table lookup
    # ========================================================================
    print("\n[Step 2] Building Router Query File...")
    _t2 = time.perf_counter()
    
    semantic_router = {}
    for qa_idx in range(len(niat_data)):
        item = niat_data[qa_idx]
        table_id = item['table_id']
        
        semantic_router[qa_idx] = {}
        semantic_router[qa_idx]['query'] = item.get('question', '')
        semantic_router[qa_idx]['title'] = item.get('table_title', table_id) or table_id
        semantic_router[qa_idx]['table'] = niat_df_processed[table_id]  # Use table_id lookup
        semantic_router[qa_idx]['table_id'] = table_id  # Store table_id for reference
        semantic_router[qa_idx]['label'] = []
    
    router_query_file = f'{args.tmp_save_path}/router_query.pkl'
    with open(router_query_file, 'wb') as f:
        pickle.dump(semantic_router, f)
    print(f"Saved router query to {router_query_file}")
    
    timeline['Step 2 - Build Router Query'] = time.perf_counter() - _t2
    print(f"  [Timing] Step 2: {timeline['Step 2 - Build Router Query']:.2f}s")
    
    # ========================================================================
    # Step 3: Construct Database
    # KEY FIX: Build database with table_id keys
    # ========================================================================
    print("\n[Step 3] Constructing Database...")
    _t3 = time.perf_counter()
    
    # Build table_id list in consistent order
    table_id_list = list(niat_df_processed.keys())
    table_id_to_db_idx = {tid: i for i, tid in enumerate(table_id_list)}
    
    table_titles = [niat_data[0].get('table_title', tid) or tid 
                   for tid in table_id_list]  # Use first item's title or table_id
    tables_for_db = [niat_df_processed[tid] for tid in table_id_list]
    
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    print(f"Database initialized with {len(tables_for_db)} unique tables")
    
    # Store mapping for later use
    table_id_mapping = {
        'table_id_list': table_id_list,
        'table_id_to_db_idx': table_id_to_db_idx,
    }
    
    timeline['Step 3 - Construct Database'] = time.perf_counter() - _t3
    print(f"  [Timing] Step 3: {timeline['Step 3 - Construct Database']:.2f}s")

    
    # ========================================================================
    # Step 4: Router Model Inference
    # ========================================================================
    print("\n[Step 4] Router Model Inference...")
    _t4 = time.perf_counter()
    
    router_result_file = f'{args.tmp_save_path}/inference_result.pkl'
    
    if not args.skip_router:
        cmd = f"python inference_router.py --input_path {router_query_file} " \
              f"--model_path {args.router_model_path} " \
              f"--output_path {router_result_file}"
        print(f"Running: {cmd}")
        os.system(cmd)
    else:
        print("Skipping router inference (--skip_router)")
    
    # Load router results
    if os.path.exists(router_result_file):
        with open(router_result_file, 'rb') as f:
            error_analysis_row = pickle.load(f)
    else:
        # Fallback: use all methods for all samples
        print("Warning: Router result not found, using default routing (all methods)")
        error_analysis_row = {i: {'Base': 0.5, 'Select_Row': 0.5, 'Select_Column': 0.5, 
                                   'Execute_SQL': 0.5, 'RAG_20_5': 0.5} 
                             for i in range(len(niat_data))}
    
    timeline['Step 4 - Router Inference'] = time.perf_counter() - _t4
    print(f"  [Timing] Step 4: {timeline['Step 4 - Router Inference']:.2f}s")
    
    # ========================================================================
    # Step 5: Parse Router Results & Organize LLM Query List
    # KEY FIX: Use table_id for table lookup
    # ========================================================================
    print("\n[Step 5] Parsing Router Results...")
    _t5 = time.perf_counter()
    
    ranked_result = process_error_analysis_list(
        error_analysis_row, truncate=True, tau=args.tau
    )
    
    print("Router result distribution:")
    print(pd.DataFrame([str(r) for r in ranked_result.values()]).value_counts())
    
    # Organize LLM query list
    LLM_query_list = {}
    for method in ALL_LABELS:
        LLM_query_list[method] = {
            'index': [],
            'query': [],
            'qa': [],
            'table_id': []  # Add table_id tracking
        }
    
    for qa_idx in range(len(niat_data)):
        table_id = niat_data[qa_idx]['table_id']  # KEY FIX: get table_id from item
        
        for method in ALL_LABELS:
            if method in ranked_result[qa_idx]:
                LLM_query_list[method]['index'].append(qa_idx)
                LLM_query_list[method]['table_id'].append(table_id)
                
                if method == 'Select_Column':
                    prompt = build_niat_prompt_from_df(
                        niat_data, niat_df_processed[table_id], qa_idx,  # Use table_id
                        template_path='../prompts/col_select_niat.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Select_Row':
                    prompt = build_niat_prompt_from_df(
                        niat_data, niat_df_processed[table_id], qa_idx,  # Use table_id
                        template_path='../prompts/row_select_niat.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Execute_SQL':
                    prompt = build_niat_prompt_from_df(
                        niat_data, niat_df_processed[table_id], qa_idx,  # Use table_id
                        template_path='../prompts/sql_reason_niat.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].append(prompt)
    
    # Save RAG index
    np.save(f'{args.tmp_save_path}/RAG_index.npy', LLM_query_list['RAG_20_5']['index'])
    
    print(f"Query counts:")
    for method in ALL_LABELS:
        print(f"  {method}: {len(LLM_query_list[method]['index'])}")
    
    timeline['Step 5 - Parse Router & Build Queries'] = time.perf_counter() - _t5
    print(f"  [Timing] Step 5: {timeline['Step 5 - Parse Router & Build Queries']:.2f}s")
    
    # ========================================================================
    # Step 6: Execute RAG Task
    # ========================================================================
    rag_count = len(LLM_query_list['RAG_20_5']['index'])
    rag_output_file = f'{args.tmp_save_path}/Hybrid_Retrieve_output.npy'
    
    print(f"\n[Step 6] Executing RAG on {rag_count} samples...")
    _t6 = time.perf_counter()
    
    if not args.skip_rag and rag_count > 0:
        # Save NIAT queries for RAG (Hybrid_Retrieve_Update_dict.py needs them)
        niat_queries = {}
        for idx in range(len(niat_data)):
            niat_queries[idx] = niat_data[idx].get('question', f'Query for table {idx}')
        rewrite_query_file = f'{args.tmp_save_path}/rewrite_query.npy'
        np.save(rewrite_query_file, niat_queries)
        
        cmd = f"python Hybrid_Retrieve_Update_dict.py --model_path {args.embedding_model_path} " \
              f"--dataset_name {args.dataset_name} --split {args.split} " \
              f"--index_path {args.tmp_save_path}/RAG_index.npy " \
              f"--output_path {rag_output_file} " \
              f"--max_rows 50 --max_cols 10 " \
              f"--processed_df_path {preprocess_file} " \
              f"--rewrite_query_path {rewrite_query_file}"
        print(f"Running: {cmd}")
        os.system(cmd)
    
    if os.path.exists(rag_output_file):
        RAG_20_5 = np.load(rag_output_file, allow_pickle=True).item()
    else:
        print("Warning: RAG output not found, using original tables")
        # KEY FIX: Use table_id for lookup
        RAG_20_5 = {qa_idx: niat_df_processed[niat_data[qa_idx]['table_id']] 
                   for qa_idx in LLM_query_list['RAG_20_5']['index']}
    
    timeline['Step 6 - RAG'] = time.perf_counter() - _t6
    print(f"  [Timing] Step 6: {timeline['Step 6 - RAG']:.2f}s")
    
    # ========================================================================
    # Step 7: Execute LLM Queries - Select_Row, Select_Column
    # ========================================================================
    print("\n[Step 7] Executing Select_Row and Select_Column...")
    _t7 = time.perf_counter()
    
    for method in ['Select_Row', 'Select_Column']:
        if LLM_query_list[method]['query']:
            prompt_list = LLM_query_list[method]['query']
            response_list, metrics_rows[method], summaries[method] = infer_prompts(
                prompt_list,
                sample_num=args.select_sample_num,
                temperature=args.temperature,
                top_p=args.top_p,
                llm_path=args.llm_path,
                concurrency=args.llm_concurrency
            )
            LLM_query_list[method]['response'] = response_list
            print(f"  {method}: {len(response_list)} responses")
            print(f"  {method} infer summary: {summaries[method]}")
    
    timeline['Step 7 - Select Ops Generation'] = time.perf_counter() - _t7
    print(f"  [Timing] Step 7: {timeline['Step 7 - Select Ops Generation']:.2f}s")
    
    # ========================================================================
    # Step 8: Execute SQL Queries
    # ========================================================================
    print("\n[Step 8] Executing Execute_SQL...")
    _t8 = time.perf_counter()
    
    if LLM_query_list['Execute_SQL']['query']:
        prompt_list = LLM_query_list['Execute_SQL']['query']
        response_list, metrics_rows["Execute_SQL"], summaries["Execute_SQL"] = infer_prompts(
            prompt_list,
            sample_num=args.sql_sample_num,
            temperature=args.temperature,
            top_p=args.top_p,
            llm_path=args.llm_path,
            concurrency=args.llm_concurrency
        )
        LLM_query_list['Execute_SQL']['response'] = response_list
        print(f"  Execute_SQL: {len(response_list)} responses")
        print(f"  Execute_SQL infer summary: {summaries['Execute_SQL']}")
    
    timeline['Step 8 - SQL Generation'] = time.perf_counter() - _t8
    print(f"  [Timing] Step 8: {timeline['Step 8 - SQL Generation']:.2f}s")
    
    # ========================================================================
    # Step 9: SQL Parse and Execute
    # KEY FIX: Use table_id from LLM_query_list for correct table lookup
    # ========================================================================
    print("\n[Step 9] Parsing and Executing SQL...")
    _t9 = time.perf_counter()
    
    # Parse Select_Row
    print("  Parsing Select_Row SQL...")
    sub_table_list_all = {}
    filtered_tables_row = {}
    row_sql_index_list = LLM_query_list['Select_Row'].get('index', [])
    row_sql_table_ids = LLM_query_list['Select_Row'].get('table_id', [])
    row_sql_response_list = LLM_query_list['Select_Row'].get('response', [])
    
    for i in range(len(row_sql_index_list)):
        sample_num = [0, 1]
        sub_table_list = []
        qa_idx = row_sql_index_list[i]
        table_id = row_sql_table_ids[i] if row_sql_table_ids else niat_data[qa_idx]['table_id']
        
        for sample_index in sample_num:
            if sample_index >= len(row_sql_response_list[i]):
                continue
            original_text = row_sql_response_list[i][sample_index]
            table_df = niat_df_processed[table_id]  # KEY FIX: use table_id
            table_title = niat_data[qa_idx].get('table_title', table_id) or table_id
            sql = fix_sql_query(
                response_text=original_text,
                table_df=table_df,
                table_title=table_title
            )
            try:
                # Get db index for this table_id
                db_idx = table_id_mapping['table_id_to_db_idx'].get(table_id, 0)
                result = executor.sql_exec(
                    sql.replace('``', '`').replace("COUNT(*)", "*"),
                    db, table_id=db_idx
                )
                sub_table_list.append(
                    pd.DataFrame(result['rows'], columns=result['header'])
                )
            except:
                continue
        
        sub_table_list_all[qa_idx] = sub_table_list
        filtered_df = retrieve_rows_by_subtables(
            niat_df_processed[table_id], sub_table_list  # KEY FIX
        )
        if len(filtered_df) == 0:
            filtered_df = niat_df_processed[table_id]  # KEY FIX
        filtered_tables_row[qa_idx] = filtered_df
    
    # Parse Select_Column
    print("  Parsing Select_Column SQL...")
    filtered_tables = {}
    filtered_headers = {}
    col_sql_index_list = LLM_query_list['Select_Column'].get('index', [])
    col_sql_table_ids = LLM_query_list['Select_Column'].get('table_id', [])
    col_sql_response_list = LLM_query_list['Select_Column'].get('response', [])
    
    for i in range(len(col_sql_index_list)):
        qa_idx = col_sql_index_list[i]
        table_id = col_sql_table_ids[i] if col_sql_table_ids else niat_data[qa_idx]['table_id']
        input_df = niat_df_processed[table_id]  # KEY FIX: use table_id
        response_list = col_sql_response_list[i]
        assert isinstance(response_list, list)
        filtered_table, final_headers = filter_dataframe_from_responses(
            response_list, input_df, add_row_id=True
        )
        filtered_tables[qa_idx] = filtered_table
        filtered_headers[qa_idx] = final_headers
    
    # Parse Execute_SQL
    print("  Parsing Execute_SQL...")
    sample_num = 3
    sql_exec_df = {}
    valid_parse = 0
    sql_executable_count = []
    exec_sql_index_list = LLM_query_list['Execute_SQL'].get('index', [])
    exec_sql_table_ids = LLM_query_list['Execute_SQL'].get('table_id', [])
    exec_sql_response_list = LLM_query_list['Execute_SQL'].get('response', [])
    
    for i in range(len(exec_sql_index_list)):
        qa_idx = exec_sql_index_list[i]
        table_id = exec_sql_table_ids[i] if exec_sql_table_ids else niat_data[qa_idx]['table_id']
        table_df = niat_df_processed[table_id]  # KEY FIX: use table_id
        table_title = niat_data[qa_idx].get('table_title', table_id) or table_id
        
        sql_exec_df[qa_idx] = []
        for sample_ind in range(min(sample_num, len(exec_sql_response_list[i]) if exec_sql_response_list else 0)):
            original_text = exec_sql_response_list[i][sample_ind]
            sql = fix_sql_query(
                response_text=original_text,
                table_df=table_df,
                table_title=table_title
            )
            if sql != '':
                try:
                    db_idx = table_id_mapping['table_id_to_db_idx'].get(table_id, 0)
                    result = executor.sql_exec(
                        sql.replace('``', '`'), db,
                        table_id=db_idx, add_row_id=True
                    )
                    df = pd.DataFrame(result['rows'], columns=result['header'])
                except:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
            sql_exec_df[qa_idx].append(df)
            if len(df) > 0:
                valid_parse += 1
                sql_executable_count.append({
                    'id': qa_idx,
                    'table_id': table_id,  # Add table_id tracking
                    'sample_ind': sample_ind,
                    'sql': sql,
                    'table': df
                })
    
    sql_exec_df_output = merge_clean_and_format_df_dict(sql_exec_df)
    
    # Aggregate processed tables (using qa_idx as key for downstream compatibility)
    processed_table = {}
    processed_table['Base'] = {qa_idx: niat_df_processed[niat_data[qa_idx]['table_id']] 
                               for qa_idx in range(len(niat_data))}
    processed_table['Select_Row'] = filtered_tables_row
    processed_table['Select_Column'] = filtered_tables
    processed_table['RAG_20_5'] = RAG_20_5
    processed_table['Execute_SQL'] = sql_exec_df_output
    processed_table['Execute_SQL_count'] = sql_executable_count
    
    np.save(f'{args.tmp_save_path}/processed_table.npy', processed_table)
    print(f"Saved processed tables to {args.tmp_save_path}/processed_table.npy")
    
    timeline['Step 9 - SQL Parsing & Execution'] = time.perf_counter() - _t9
    print(f"  [Timing] Step 9: {timeline['Step 9 - SQL Parsing & Execution']:.2f}s")
    
    # ========================================================================
    # Step 10: Check Model Iteration
    # ========================================================================
    print("\n[Step 10] Running Check Model...")
    _t10 = time.perf_counter()
    
    # Initialize Check Model Data Sequence
    Check_Model_Data_Sequence = {}
    for key in ranked_result.keys():
        start_sequence = ranked_result[key]
        Check_Model_Data_Sequence[key] = {
            'id': key,
            'Sequence': start_sequence,
            'Terminated': start_sequence == ['Base'] or start_sequence == ['Execute_SQL'],
            'Check_Status': False,
            'Check_Score': 0.0
        }
    
    for key in Check_Model_Data_Sequence.keys():
        data_entry = Prepare_Data_for_Operator_Sequence_NIAT(
            key, Check_Model_Data_Sequence[key]['Sequence'],
            niat_data, processed_table
        )
        Check_Model_Data_Sequence[key]['data_entry'] = data_entry
    
    # Load reranker model
    print("  Loading reranker model...")
    reranker_model = FlagReranker(
        args.check_model_path,
        use_fp16=True,
        devices=[0]
    )
    
    # Iterative check (3 loops)
    print("  Running iterative check (3 rounds)...")
    check_tau = args.check_tau
    for loop in range(3):
        print(f"    Round {loop + 1}/3...")
        updated_data = batch_rerank_scores(
            reranker_model, Check_Model_Data_Sequence, batch_size=16
        )
        Check_Model_Data_Sequence = updated_data
        
        for key in Check_Model_Data_Sequence.keys():
            if Check_Model_Data_Sequence[key]['Terminated']:
                continue
            if Check_Model_Data_Sequence[key]['Check_Status']:
                if Check_Model_Data_Sequence[key]['Check_Score'] >= check_tau:
                    Check_Model_Data_Sequence[key]['Terminated'] = True
                else:
                    Check_Model_Data_Sequence[key]['Terminated'] = False
                    Check_Model_Data_Sequence[key]['Check_Status'] = False
                    Check_Model_Data_Sequence[key]['Check_Score'] = 0.0
                    current_sequence = Check_Model_Data_Sequence[key]['Sequence']
                    ROLLBACK_seq, terminated_flag = ROLLBACK(current_sequence)
                    Check_Model_Data_Sequence[key]['Sequence'] = ROLLBACK_seq
                    Check_Model_Data_Sequence[key]['Terminated'] = terminated_flag
                    if not terminated_flag:
                        data_entry = Prepare_Data_for_Operator_Sequence_NIAT(
                            key, ROLLBACK_seq, niat_data, processed_table
                        )
                        Check_Model_Data_Sequence[key]['data_entry'] = data_entry
    
    np.save(f'{args.tmp_save_path}/Check_Model_Data_Sequence.npy', Check_Model_Data_Sequence)
    
    timeline['Step 10 - Check Model Iteration'] = time.perf_counter() - _t10
    print(f"  [Timing] Step 10: {timeline['Step 10 - Check Model Iteration']:.2f}s")
    
    # Cleanup reranker
    import gc
    import torch
    del reranker_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================================================
    # Step 11: Add Missing Execute_SQL
    # ========================================================================
    print("\n[Step 11] Adding Missing Execute_SQL...")
    _t11 = time.perf_counter()
    
    SQL_list_final = []
    for index in range(len(niat_data)):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        if sequence == [] or 'Execute_SQL' in sequence:
            SQL_list_final.append(index)
    
    add_sql_list = list(
        set(SQL_list_final) - set(LLM_query_list['Execute_SQL']['index'])
    )
    
    print(f"  Adding {len(add_sql_list)} missing SQL queries...")
    if add_sql_list:
        add_sql_query_list = []
        for index in add_sql_list:
            prompt = build_niat_prompt_from_df(
                niat_data, niat_df_processed[index], index,
                template_path='../prompts/sql_reason_niat.txt',
                processed=True
            )
            add_sql_query_list.append(prompt)
        
        add_sql_response_list, metrics_rows["Add_SQL"], summaries["Add_SQL"] = infer_prompts(
            add_sql_query_list,
            sample_num=args.sql_sample_num,
            temperature=args.temperature,
            top_p=args.top_p,
            llm_path=args.llm_path,
            concurrency=args.llm_concurrency
        )
        print(f"  Add SQL generation infer summary: {summaries['Add_SQL']}")
        
        # Parse additional SQL
        for i in range(len(add_sql_list)):
            index = add_sql_list[i]
            sql_exec_df[index] = []
            for sample_ind in range(args.sql_sample_num):
                if sample_ind >= len(add_sql_response_list[i]):
                    continue
                original_text = add_sql_response_list[i][sample_ind]
                sql = fix_sql_query(
                    response_text=original_text,
                    table_df=niat_df_processed[index],
                    table_title=table_titles[index]
                )
                if sql != '':
                    try:
                        result = executor.sql_exec(
                            sql.replace('``', '`'), db,
                            table_id=index, add_row_id=True
                        )
                        df = pd.DataFrame(result['rows'], columns=result['header'])
                    except:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
                sql_exec_df[index].append(df)
        
        sql_exec_df_output_new = merge_clean_and_format_df_dict(sql_exec_df)
        for index in sql_exec_df_output_new.keys():
            sql_exec_df_output[index] = sql_exec_df_output_new[index]
        processed_table['Execute_SQL'] = sql_exec_df_output
    
    np.save(f'{args.tmp_save_path}/processed_table.npy', processed_table)
    
    timeline['Step 11 - Add Missing SQL'] = time.perf_counter() - _t11
    print(f"  [Timing] Step 11: {timeline['Step 11 - Add Missing SQL']:.2f}s")
    
    # ========================================================================
    # Step 12: Generate Final QA Prompts
    # ========================================================================
    print("\n[Step 12] Generating Final QA Prompts...")
    _t12 = time.perf_counter()
    
    prompt_list = []
    for index in range(len(niat_data)):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        prompt = build_niat_prompt_from_df(
            niat_data,
            Check_Model_Data_Sequence[index]['data_entry']['table'],
            index,
            template_path='../prompts/text_reason_niat.txt',
            processed=True
        )
        if sequence == [] or 'Execute_SQL' in sequence:
            if index in processed_table['Execute_SQL']:
                evidence = table_to_str_sql(processed_table['Execute_SQL'][index])
                prompt = prompt + evidence
        prompt_list.append(prompt)
    
    timeline['Step 12 - Generate Final QA Prompts'] = time.perf_counter() - _t12
    print(f"  [Timing] Step 12: {timeline['Step 12 - Generate Final QA Prompts']:.2f}s")
    
    # ========================================================================
    # Step 13: Execute Final QA and Evaluate
    # ========================================================================
    print("\n[Step 13] Executing Final QA...")
    _t13 = time.perf_counter()
    
    qa_final, metrics_rows["Final_QA"], summaries["Final_QA"] = infer_prompts(
        prompt_list,
        sample_num=1,
        temperature=0,
        top_p=1,
        llm_path=args.llm_path,
        concurrency=max(4, min(8, args.llm_concurrency))
    )
    print(f"  Final QA generation infer summary: {summaries['Final_QA']}")
    
    # Create result dataframe
    niat_df = pd.DataFrame(niat_data)
    niat_df['instruction'] = prompt_list
    niat_df['predict'] = [str(s) for s in qa_final]
    
    timeline['Step 13 - Final QA Generation'] = time.perf_counter() - _t13
    print(f"  [Timing] Step 13: {timeline['Step 13 - Final QA Generation']:.2f}s")
    
    # Evaluate using NIAT-specific evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    # Build samples for evaluation
    all_samples = []
    for i in range(len(niat_data)):
        sample = {
            'chain': [{'parameter_and_conf': [(qa_final[i][0] if isinstance(qa_final[i], list) else qa_final[i], 1.0)]}],
            'answer': niat_data[i].get('answer', '')
        }
        all_samples.append(sample)
    
    acc_all = niat_match_func_for_samples(all_samples, strategy="top") * 100
    print(f"Accuracy (NIAT EM): {acc_all:.2f}%")
    
    total_time = time.perf_counter() - overall_start
    
    # Save results
    niat_df.to_csv(f'{args.tmp_save_path}/final_results.csv', index=False)
    print(f"Saved results to {args.tmp_save_path}/final_results.csv")
    
    # Save evaluation results
    with open(f'{args.tmp_save_path}/evaluation_results.json', 'w') as f:
        json.dump({
            'accuracy': acc_all,
            'total_samples': len(niat_data)
        }, f, indent=2)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    metrics_rows_f = f"{args.tmp_save_path}/metrics_n{args.first_n}_p{args.llm_concurrency}.json"
    summaries_f = f"{args.tmp_save_path}/summaries_n{args.first_n}_p{args.llm_concurrency}.json"
    with open(metrics_rows_f, 'w') as f:
        json.dump(metrics_rows, f, indent=2)
    with open(summaries_f, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    total_latency = sum(v.get('batch_dur', 0) for v in summaries.values())
    total_prompt_tokens = sum(v.get('total_prompt_tokens', 0) for v in summaries.values())
    total_completion_tokens = sum(v.get('total_completion_tokens', 0) for v in summaries.values())
    
    print(f"\n{'=' * 80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    print(f"Accuracy: {acc_all:.2f}%")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nTiming Breakdown:")
    print(f"{'-' * 80}")
    for step_name, duration in timeline.items():
        percentage = (duration / total_time) * 100
        print(f"  {step_name:<45} {duration:>8.2f}s ({percentage:>5.1f}%)")
    print(f"{'-' * 80}")
    print(f"Results saved to: {args.tmp_save_path}")
    print(f"{'=' * 80}\n")
    
    # Save timeline
    timeline_summary = {
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'accuracy': acc_all,
        'steps': timeline
    }
    with open(f'{args.tmp_save_path}/timing_summary_n{args.first_n}_p{args.llm_concurrency}.json', 'w') as f:
        json.dump(timeline_summary, f, indent=2)


if __name__ == "__main__":
    main()
