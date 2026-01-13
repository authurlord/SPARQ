#!/usr/bin/env python3
"""
Complete Pipeline for NIAT Dataset (Nested/Hierarchical Tables)
Adapted from run_full_pipeline_wikitq.py

This script provides full LLM-based table QA pipeline for NIAT dataset:
1. Loading and preprocessing NIAT nested tables (with flatten preprocessing)
2. Building database and executing SQL queries
3. LLM inference via vLLM or async API for:
   - Column selection
   - Row selection
   - SQL execution
   - Final QA
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from utils.multi_db_v2 import NeuralDB, Executor
from utils.prompt_generate import (
    format_table_prompt, fix_sql_query, filter_dataframe_from_responses
)
from utils.schedule_utils import (
    table_to_str, table_to_str_sql,
    merge_clean_and_format_df_dict,
)

# Optional imports for full pipeline (may not be available in demo mode)
try:
    from utils.async_llm import infer_prompts
    ASYNC_LLM_AVAILABLE = True
except ImportError:
    ASYNC_LLM_AVAILABLE = False
    print("Warning: async_llm not available, LLM inference will be skipped")

try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    print("Warning: FlagEmbedding not available, reranking will be skipped")

# vLLM imports (optional)
try:
    import multiprocessing as mp
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
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, use --demo_mode or --use_api")


def parse_args():
    parser = argparse.ArgumentParser(description="NIAT Full Pipeline")
    
    # Model paths
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
    
    # Pipeline parameters
    parser.add_argument('--tau', type=float, default=0.82, help='Router threshold')
    parser.add_argument('--check_tau', type=float, default=0.8, help='Check model threshold')
    parser.add_argument('--n_parallel', type=int, default=4, help='Number of parallel workers')
    
    # vLLM parameters
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--max_model_len', type=int, default=23000,
                       help='Maximum model length')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                       help='GPU memory utilization')
    parser.add_argument('--max_num_seqs', type=int, default=256,
                       help='Maximum number of sequences')
    
    # Sampling parameters
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
    
    # Execution control
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--skip_llm', action='store_true',
                       help='Skip LLM inference (for testing data flow)')
    parser.add_argument('--first_n', type=int, default=-1,
                       help='Only process first N samples (-1 for all)')
    parser.add_argument('--demo_mode', action='store_true',
                       help='Demo mode: skip LLM inference, only test data pipeline')
    parser.add_argument('--use_api', action='store_true',
                       help='Use async API instead of local vLLM')
    
    return parser.parse_args()


# ============================================================================
# vLLM Functions (same as wikitq pipeline)
# ============================================================================

def response_vllm(llm, tokenizer, all_instructions: List[str], sample_num: int, 
                 temperature: float = 0.7, top_p: float = 0.8) -> List[List[str]]:
    """
    Generate responses using vLLM - exactly as in wikitq pipeline
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not available. Use --demo_mode or --use_api")
    
    text_all = []
    for prompt in tqdm(all_instructions, desc="Formatting Prompts"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        text_all.append(text)
    
    if sample_num == 1:
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, presence_penalty=1.0)
    else:
        sampling_params = SamplingParams(
            n=sample_num, temperature=temperature, top_p=top_p,
            top_k=20, min_p=0, max_tokens=2048, presence_penalty=1.0
        )
    
    outputs = llm.generate(text_all, sampling_params)
    generation_list = []
    for output in tqdm(outputs, desc="Processing Outputs"):
        generated_text = [o.text for o in output.outputs]
        generation_list.append(generated_text)
    
    return generation_list


def init_llm_and_tokenizer(args):
    """Initialize vLLM and tokenizer - exactly as in wikitq pipeline"""
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not available. Use --demo_mode or --use_api")
    
    llm = LLM(
        model=args.llm_path,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=True,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    print("vLLM and tokenizer initialized")
    return llm, tokenizer


# ============================================================================
# NIAT-specific Functions
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
        
        # Make column name SQL-safe (lowercase, replace spaces)
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
    """
    Flatten a hierarchical/nested table by forward-filling empty cells.
    Also handles duplicate column names by adding suffixes.
    """
    if not table_rows or len(table_rows) < 2:
        if len(table_rows) == 1:
            header = make_unique_columns(table_rows[0])
            return pd.DataFrame(columns=header)
        return pd.DataFrame()
    
    header = make_unique_columns(table_rows[0])
    data_rows = table_rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    
    # For hierarchical tables, forward-fill left-most grouping columns
    if table_structure == "hierarchical":
        for col_idx in range(min(2, len(df.columns))):
            col_name = df.columns[col_idx]
            col_values = df[col_name].values
            
            # Count empty-after-non-empty patterns
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


def build_niat_prompt(item: Dict, df: pd.DataFrame, 
                      template_path: str = '../prompts/col_select_niat.txt') -> str:
    """Build a prompt for NIAT item using the specified template."""
    template_full_path = os.path.join(os.path.dirname(__file__), template_path)
    if os.path.exists(template_full_path):
        with open(template_full_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = ""
    
    table_title = item.get('table_title', 'Table')
    if not table_title:
        table_title = item.get('table_id', 'Table')
    
    question = item.get('question', '')
    prompt = format_table_prompt(table_title, df, question)
    
    if template:
        prompt = template + "\n\n" + prompt
    
    return prompt


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    args = parse_args()
    
    print("=" * 80)
    print("NIAT Full Pipeline")
    print("=" * 80)
    print(f"Mode: {'Demo (no LLM)' if args.demo_mode else ('API' if args.use_api else 'vLLM')}")
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    timeline = {}
    overall_start = time.perf_counter()
    
    # Initialize LLM if not in demo mode
    llm, tokenizer = None, None
    if not args.demo_mode and not args.use_api and not args.skip_llm:
        if VLLM_AVAILABLE:
            print("\n[Init] Initializing vLLM...")
            llm, tokenizer = init_llm_and_tokenizer(args)
        else:
            print("Warning: vLLM not available, switching to demo mode")
            args.demo_mode = True
    
    # ========================================================================
    # Step 1: Load and Preprocess NIAT Data
    # ========================================================================
    print("\n[Step 1] Loading NIAT Dataset...")
    _t1 = time.perf_counter()
    
    niat_data = load_niat_dataset(args.niat_json_path, args.first_n)
    
    # Process tables with flattening
    print("Processing nested tables with flattening...")
    processed_tables = {}
    for idx, item in enumerate(tqdm(niat_data, desc="Flattening tables")):
        table_rows = item.get('table_rows', [])
        table_structure = item.get('table_structure', 'vertical')
        df = flatten_hierarchical_table(table_rows, table_structure)
        processed_tables[idx] = df
    
    # Analyze structure distribution
    structure_counts = {}
    for item in niat_data:
        struct = item.get('table_structure', 'unknown')
        structure_counts[struct] = structure_counts.get(struct, 0) + 1
    print(f"Table structure distribution: {structure_counts}")
    
    timeline['Step 1 - Data Loading'] = time.perf_counter() - _t1
    
    # ========================================================================
    # Step 2: Construct Database
    # ========================================================================
    print("\n[Step 2] Constructing Database...")
    _t2 = time.perf_counter()
    
    table_titles = [item.get('table_title', f"Table_{i}") or f"Table_{i}" 
                   for i, item in enumerate(niat_data)]
    tables_for_db = [processed_tables[i] for i in range(len(niat_data))]
    
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    print(f"Database initialized with {len(tables_for_db)} tables")
    
    timeline['Step 2 - Database Construction'] = time.perf_counter() - _t2
    
    # ========================================================================
    # Step 3: Build LLM Query Lists
    # ========================================================================
    print("\n[Step 3] Building Query Lists...")
    _t3 = time.perf_counter()
    
    ALL_LABELS = ['Select_Column', 'Select_Row', 'Execute_SQL']
    LLM_query_list = {method: {'index': [], 'query': [], 'response': []} for method in ALL_LABELS}
    
    for idx, item in enumerate(tqdm(niat_data, desc="Building queries")):
        df = processed_tables[idx]
        
        # Select_Column
        col_prompt = build_niat_prompt(item, df, '../prompts/col_select_niat.txt')
        LLM_query_list['Select_Column']['index'].append(idx)
        LLM_query_list['Select_Column']['query'].append(col_prompt)
        
        # Select_Row
        row_prompt = build_niat_prompt(item, df, '../prompts/row_select_niat.txt')
        LLM_query_list['Select_Row']['index'].append(idx)
        LLM_query_list['Select_Row']['query'].append(row_prompt)
        
        # Execute_SQL
        sql_prompt = build_niat_prompt(item, df, '../prompts/sql_reason_niat.txt')
        LLM_query_list['Execute_SQL']['index'].append(idx)
        LLM_query_list['Execute_SQL']['query'].append(sql_prompt)
    
    print(f"Query counts per method:")
    for method in ALL_LABELS:
        print(f"  {method}: {len(LLM_query_list[method]['query'])}")
    
    timeline['Step 3 - Query Building'] = time.perf_counter() - _t3
    
    # ========================================================================
    # Step 4: LLM Inference (Skip in demo mode)
    # ========================================================================
    if not args.demo_mode and not args.skip_llm:
        print("\n[Step 4] Running LLM Inference...")
        _t4 = time.perf_counter()
        
        for method in ['Select_Column', 'Select_Row']:
            if LLM_query_list[method]['query']:
                prompt_list = LLM_query_list[method]['query']
                print(f"  Processing {method}: {len(prompt_list)} prompts...")
                
                if args.use_api and ASYNC_LLM_AVAILABLE:
                    response_list, _, _ = infer_prompts(
                        prompt_list,
                        sample_num=args.select_sample_num,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        llm_path=args.llm_path,
                        concurrency=args.llm_concurrency
                    )
                elif llm is not None:
                    response_list = response_vllm(
                        llm, tokenizer, prompt_list,
                        sample_num=args.select_sample_num,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )
                else:
                    response_list = [[""] * args.select_sample_num for _ in prompt_list]
                
                LLM_query_list[method]['response'] = response_list
        
        # Execute_SQL
        if LLM_query_list['Execute_SQL']['query']:
            prompt_list = LLM_query_list['Execute_SQL']['query']
            print(f"  Processing Execute_SQL: {len(prompt_list)} prompts...")
            
            if args.use_api and ASYNC_LLM_AVAILABLE:
                response_list, _, _ = infer_prompts(
                    prompt_list,
                    sample_num=args.sql_sample_num,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    llm_path=args.llm_path,
                    concurrency=args.llm_concurrency
                )
            elif llm is not None:
                response_list = response_vllm(
                    llm, tokenizer, prompt_list,
                    sample_num=args.sql_sample_num,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
            else:
                response_list = [[""] * args.sql_sample_num for _ in prompt_list]
            
            LLM_query_list['Execute_SQL']['response'] = response_list
        
        timeline['Step 4 - LLM Inference'] = time.perf_counter() - _t4
    else:
        print("\n[Step 4] LLM Inference (SKIPPED - demo mode)")
        timeline['Step 4 - LLM Inference'] = 0
    
    # ========================================================================
    # Step 5: SQL Parsing and Execution
    # ========================================================================
    print("\n[Step 5] SQL Parsing and Execution...")
    _t5 = time.perf_counter()
    
    sql_results = {'success': 0, 'failed': 0, 'details': []}
    
    # Test SQL execution on sample queries
    for idx in range(min(10, len(niat_data))):
        df = processed_tables[idx]
        
        test_queries = [
            "SELECT * FROM w LIMIT 5",
            "SELECT COUNT(*) FROM w",
        ]
        
        if len(df.columns) > 0:
            col = df.columns[0]
            test_queries.append(f"SELECT `{col}` FROM w")
        
        for sql in test_queries:
            try:
                result = executor.sql_exec(sql, db, table_id=idx, verbose=False)
                sql_results['success'] += 1
            except Exception as e:
                sql_results['failed'] += 1
    
    print(f"  SQL Test: {sql_results['success']} success, {sql_results['failed']} failed")
    
    timeline['Step 5 - SQL Execution'] = time.perf_counter() - _t5
    
    # ========================================================================
    # Step 6: Build Final QA Prompts
    # ========================================================================
    print("\n[Step 6] Building Final QA Prompts...")
    _t6 = time.perf_counter()
    
    final_prompts = []
    for idx, item in enumerate(tqdm(niat_data, desc="Building QA prompts")):
        df = processed_tables[idx]
        qa_prompt = build_niat_prompt(item, df, '../prompts/text_reason_niat.txt')
        final_prompts.append({
            'index': idx,
            'prompt': qa_prompt,
            'question': item.get('question', ''),
            'answer': item.get('answer', '')
        })
    
    timeline['Step 6 - Final QA Prompts'] = time.perf_counter() - _t6
    
    # ========================================================================
    # Step 7: Save Results
    # ========================================================================
    print("\n[Step 7] Saving Results...")
    _t7 = time.perf_counter()
    
    # Save processed tables
    np.save(os.path.join(args.tmp_save_path, 'niat_df_processed.npy'), processed_tables)
    
    # Save query lists
    with open(os.path.join(args.tmp_save_path, 'llm_query_list.pkl'), 'wb') as f:
        pickle.dump(LLM_query_list, f)
    
    # Save prompts sample
    prompts_sample = {
        'Select_Column': LLM_query_list['Select_Column']['query'][:3],
        'Select_Row': LLM_query_list['Select_Row']['query'][:3],
        'Execute_SQL': LLM_query_list['Execute_SQL']['query'][:3],
        'Final_QA': [p['prompt'] for p in final_prompts[:3]]
    }
    with open(os.path.join(args.tmp_save_path, 'prompts_sample.json'), 'w', encoding='utf-8') as f:
        json.dump(prompts_sample, f, ensure_ascii=False, indent=2)
    
    # Save SQL results
    with open(os.path.join(args.tmp_save_path, 'sql_test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(sql_results, f, ensure_ascii=False, indent=2)
    
    timeline['Step 7 - Save Results'] = time.perf_counter() - _t7
    
    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.perf_counter() - overall_start
    
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Mode: {'Demo' if args.demo_mode else ('API' if args.use_api else 'vLLM')}")
    print(f"Total samples: {len(niat_data)}")
    print(f"Table structures: {structure_counts}")
    print(f"\nSQL Test: {sql_results['success']} success, {sql_results['failed']} failed")
    print(f"\nTiming:")
    for step, duration in timeline.items():
        print(f"  {step}: {duration:.2f}s")
    print(f"\nTotal: {total_time:.2f}s")
    print(f"Results: {args.tmp_save_path}")
    
    # Save timeline
    with open(os.path.join(args.tmp_save_path, 'timeline.json'), 'w') as f:
        json.dump(timeline, f, indent=2)
    
    # Cleanup
    if llm is not None:
        del llm
        del tokenizer
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    del db
    
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
