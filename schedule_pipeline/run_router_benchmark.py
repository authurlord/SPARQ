#!/usr/bin/env python3
"""
Router Model Benchmark Pipeline for WikiTQ
Iterates over different router models to benchmark their performance.

Based on run_full_pipeline_wikitq.py structure.
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
    load_data_split, table_to_str, table_to_str_sql,
    find_intersection_and_add_row_id, Prepare_Data_for_Operator_Sequence,
    format_document, batch_rerank_scores, ROLLBACK,
    merge_clean_and_format_df_dict, retrieve_rows_by_subtables,
    process_error_analysis_list
)
from utils.evaluator import Evaluator
from utils.prompt_generate import (
    build_wikitq_prompt_from_df, evaluate_predictions,
    filter_dataframe_from_responses, fix_sql_query,
    match_subtables, retrieve_rows_by_subtables
)
from utils.multi_db_v2 import NeuralDB, Executor
from FlagEmbedding import FlagReranker

import multiprocessing as mp

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Router Model Benchmark Pipeline for WikiTQ")
    
    # Model paths
    parser.add_argument('--llm_path', type=str, 
                       default='/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/',
                       help='Path to LLM model (constant)')
    parser.add_argument('--llm_name', type=str, default='qwen2.5-7b-instruct',
                       help='Model name for API')
    parser.add_argument('--embedding_model_path', type=str,
                       default='/home/yanmy/models/bge-m3',
                       help='Path to embedding model')
    parser.add_argument('--router_models_dir', type=str,
                       default='../models',
                       help='Directory containing router models to benchmark')
    parser.add_argument('--router_model_pattern', type=str,
                       default='*',
                       help='Glob pattern to filter router models')
    parser.add_argument('--check_model_path', type=str,
                       default='/home/yanmy/HybridRAG/H-STAR/check/wikitq',
                       help='Path to check model (constant)')
    
    # Dataset parameters - WikiTQ
    parser.add_argument('--dataset_name', type=str, default='wikitq',
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split')
    parser.add_argument('--preprocess_file', type=str,
                       default='datasets/schedule_test/wikitq/wikitq_df_processed.npy',
                       help='Path to preprocessed WikiTQ data')
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
    parser.add_argument('--api_key', type=str, default='api-key')
    
    # Skip options
    parser.add_argument('--skip_preprocess', action='store_true')
    
    return parser.parse_args()


def discover_router_models(base_dir: str, pattern: str) -> List[str]:
    """Discover router models in the given directory."""
    search_path = os.path.join(base_dir, pattern)
    candidates = glob(search_path)
    
    # Filter to only directories that look like model directories
    models = []
    for path in candidates:
        if os.path.isdir(path):
            # Check if it contains model files
            files = os.listdir(path)
            if any(f.endswith(('.bin', '.safetensors', 'config.json')) for f in files):
                models.append(path)
    
    return sorted(models)


def run_single_router(args, dataset, wikitq_df_processed: Dict[int, pd.DataFrame],
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
    for idx in range(len(dataset)):
        semantic_router[idx] = {
            'query': dataset[idx]['question'],
            'title': dataset[idx]['table']['page_title'],
            'table': wikitq_df_processed[idx],
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
    print(f"    Running: {cmd}")
    os.system(cmd)
    
    if os.path.exists(router_result_file):
        with open(router_result_file, 'rb') as f:
            error_analysis_row = pickle.load(f)
    else:
        print("  Warning: Router result not found, using default routing")
        error_analysis_row = {i: {'Base': 0.5, 'Select_Row': 0.5, 'Select_Column': 0.5, 
                                  'Execute_SQL': 0.5, 'RAG_20_5': 0.5} 
                             for i in range(len(dataset))}
    
    ranked_result = process_error_analysis_list(error_analysis_row, truncate=True, tau=args.tau)
    
    # Step 3: Build LLM Queries based on routing
    print("  [3] Building LLM Queries...")
    LLM_query_list = {method: {'index': [], 'query': []} for method in ALL_LABELS}
    
    for idx in range(len(dataset)):
        for method in ALL_LABELS:
            if method in ranked_result[idx]:
                LLM_query_list[method]['index'].append(idx)
                
                if method == 'Select_Column':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[idx], idx,
                        template_path='../prompts/col_select_sql.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Select_Row':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[idx], idx,
                        template_path='../prompts/row_select_sql.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
                elif method == 'Execute_SQL':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[idx], idx,
                        template_path='../prompts/sql_reason_wtq.txt'
                    )
                    LLM_query_list[method]['query'].append(prompt)
    
    # Step 4: LLM Inference for each method
    print("  [4] LLM Inference...")
    llm_results = {}
    
    for method in ['Select_Row', 'Select_Column', 'Execute_SQL']:
        if len(LLM_query_list[method]['query']) > 0:
            print(f"    {method}: {len(LLM_query_list[method]['query'])} queries")
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
    
    # Step 5: Execute SQL and collect results
    print("  [5] Executing SQL...")
    processed_table = {'Base': wikitq_df_processed}
    
    Execute_SQL_count = []
    for idx in llm_results.get('Execute_SQL', {}):
        responses = llm_results['Execute_SQL'][idx]
        df = wikitq_df_processed[idx]
        
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
    for idx in range(len(dataset)):
        prompt = build_wikitq_prompt_from_df(
            dataset, wikitq_df_processed[idx], idx,
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
    
    # Step 7: Evaluate using WikiTQ metrics
    print("  [7] Evaluating...")
    preds = []
    golds = []
    
    for idx, item in enumerate(dataset):
        pred = final_responses[idx][0] if isinstance(final_responses[idx], list) else final_responses[idx]
        gold = item.get('answers', item.get('answer', ['']))
        if isinstance(gold, list):
            gold = gold[0] if gold else ''
        preds.append(str(pred))
        golds.append(str(gold))
    
    # Evaluate predictions using WikiTQ evaluator
    evaluator = Evaluator()
    eval_results = evaluate_predictions(preds, golds, evaluator)
    
    results = {
        'router_model': router_name,
        'router_path': router_model_path,
        'num_samples': len(dataset),
        'exact_match': eval_results.get('exact_match', 0),
        'accuracy': eval_results.get('accuracy', 0),
        'routing_distribution': pd.DataFrame([str(r) for r in ranked_result.values()]).value_counts().to_dict()
    }
    
    # Save results
    with open(f'{output_path}/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Accuracy: {results['accuracy']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Router Model Benchmark Pipeline (WikiTQ)")
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
    
    # Load WikiTQ data
    print("\nLoading WikiTQ dataset...")
    dataset = load_data_split(args.dataset_name, args.split)
    print(f"Loaded {len(dataset)} samples from {args.dataset_name}/{args.split}")
    
    if args.first_n > 0:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))
        print(f"Processing only first {args.first_n} samples")
    
    # Load preprocessed tables
    if os.path.exists(args.preprocess_file):
        wikitq_df_processed = np.load(args.preprocess_file, allow_pickle=True).item()
        print(f"Loaded preprocessed tables from {args.preprocess_file}")
    else:
        print(f"Preprocessed file not found: {args.preprocess_file}")
        print("Please run convert_df_type_parallel.py first")
        return
    
    # Construct database
    print("\nConstructing database...")
    table_titles = [dataset[i]['table']['page_title'] for i in range(len(dataset))]
    tables_for_db = [wikitq_df_processed[i] for i in range(len(dataset))]
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    print("Database initialized")
    
    # Run benchmark for each router model
    all_results = {}
    
    for router_model_path in router_models:
        router_name = os.path.basename(router_model_path)
        output_path = f'{args.tmp_save_path}/{router_name}'
        
        try:
            results = run_single_router(
                args, dataset, wikitq_df_processed,
                db, executor, router_model_path, output_path
            )
            all_results[router_name] = results
        except Exception as e:
            import traceback
            print(f"ERROR testing {router_name}: {e}")
            traceback.print_exc()
            all_results[router_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Router Model':<40} {'Accuracy':<10}")
    print("-" * 60)
    
    for router_name, results in sorted(all_results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
        if 'error' in results:
            print(f"{router_name:<40} {'ERROR':<10} {results['error'][:30]}")
        else:
            print(f"{router_name:<40} {results['accuracy']:.4f}")
    
    # Save summary
    summary_path = f'{args.tmp_save_path}/summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
