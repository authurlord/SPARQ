#!/usr/bin/env python3
"""
Simplified TableBench SQL Pipeline with Iterative Retry (Batch Version)
- Only uses Execute_SQL (no router, no RAG, no check)
- Iterative retry: if SQL execution fails, append error and retry (max 3 iterations)
- Batch processing: uses infer_prompts for batch queries instead of one-by-one
- SQL execution timeout: configurable timeout to skip slow queries
- Reports SQL execution success rate
- Sample 3 SQL queries per question
- Test on random 50 samples
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from datetime import datetime
import re
from collections import Counter
import signal
from contextlib import contextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_llm import infer_prompts
from utils.schedule_utils import table_to_str_sql
from utils.evaluator import evaluate_tablebench_predictions
from utils.prompt_generate import fix_sql_query
from utils.multi_db_v2 import NeuralDB, Executor

import multiprocessing as mp

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class TimeoutException(Exception):
    """Exception raised when SQL execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time using SIGALRM."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"SQL execution timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified SQL Pipeline with Iterative Retry (Batch)")
    
    parser.add_argument('--llm_path', type=str, 
                       default='/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/', help='Path to LLM model')
    parser.add_argument('--llm_name', type=str, 
                       default='qwen3-4b', help='Model name registered in vLLM API server')
    parser.add_argument('--dataset_name', type=str, default='tablebench', help='Dataset name')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/tablebench_sql_iterative',
                       help='Temporary save path')
    parser.add_argument('--tablebench_jsonl_path', type=str,
                       default='../datasets/TableBench/TableBench.jsonl',
                       help='Path to TableBench JSONL file')
    
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--llm_concurrency', type=int, default=32, help='Max concurrent requests')
    
    parser.add_argument('--sql_sample_num', type=int, default=3, help='Number of SQL samples per question')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8, help='Sampling top_p')
    parser.add_argument('--max_iterations', type=int, default=3, help='Max retry iterations for failed SQL')
    parser.add_argument('--sql_timeout', type=int, default=30, help='SQL execution timeout in seconds')
    
    parser.add_argument('--first_n', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--random_sample', action='store_true', help='Randomly sample instead of first N')
    parser.add_argument('--use_api', action='store_true', help='Use async API')
    
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1", help='vLLM API Base URL')
    parser.add_argument('--api_key', type=str, default="api-key-qwen3", help='vLLM API Key')
    
    return parser.parse_args()


def load_tablebench_dataset(jsonl_path: str, first_n: int = -1, random_sample: bool = False) -> List[Dict]:
    """Load TableBench dataset from JSONL file."""
    print(f"Loading TableBench dataset from {jsonl_path}...")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    
    print(f"Loaded {len(data)} total samples")
    
    if first_n > 0:
        if random_sample:
            print(f"Randomly sampling {first_n} samples...")
            data = random.sample(data, min(first_n, len(data)))
        else:
            print(f"Taking first {first_n} samples...")
            data = data[:first_n]
    
    print(f"Using {len(data)} samples for testing")
    return data


def make_unique_columns(columns: List[str]) -> List[str]:
    """Make column names unique."""
    seen = {}
    result = []
    for col in columns:
        col_str = str(col).strip() if col else "unnamed"
        if not col_str: col_str = "unnamed"
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
    """Convert TableBench table to DataFrame."""
    columns = item['table']['columns']
    data = item['table']['data']
    unique_columns = make_unique_columns(columns)
    return pd.DataFrame(data, columns=unique_columns)


def build_sql_prompt(item: Dict, df: pd.DataFrame, template_path: str, 
                     error_msg: str = None, iteration: int = 0) -> str:
    """Build SQL generation prompt with optional error feedback."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Add row_id if not present
    temp_df = df.copy()
    if 'row_id' not in temp_df.columns:
        temp_df.insert(0, 'row_id', temp_df.index)
    table_str = temp_df.to_string(index=False)
    
    columns = df.columns.tolist()
    all_cols = ['row_id'] + columns if 'row_id' not in columns else columns
    
    table_title = item.get('id', 'Table')
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(table_title))
    
    schema_cols = [f"`{col}` text" for col in columns]
    if 'row_id' not in columns:
        schema_cols.insert(0, "row_id int")
    col_defs = ",\n\t".join(schema_cols)
    schema = f"CREATE TABLE {table_name}(\n\t{col_defs})"
    
    question = item.get('question', '')
    
    # Add error feedback for retry iterations
    error_feedback = ""
    if error_msg and iteration > 0:
        error_feedback = f"\n\n[Previous Attempt Failed - Iteration {iteration}]\nError: {error_msg}\nPlease generate a corrected SQL query that avoids this error.\n"
    
    input_section = f"""
<input>
{schema}
/*
SELECT * FROM w;
{table_str}
*/
columns: {all_cols}
Q: {question}
{error_feedback}
<output>"""
    
    prompt = template.strip() + "\n" + input_section
    return prompt


def execute_sql_with_timeout(executor: Executor, sql: str, db: NeuralDB, 
                             table_id: int, timeout: int) -> Tuple[bool, Any, str]:
    """
    Execute SQL with timeout.
    Returns: (success, result_df, error_msg)
    """
    try:
        with time_limit(timeout):
            result = executor.sql_exec(
                sql.replace('``', '`'), db,
                table_id=table_id, add_row_id=True
            )
            result_df = pd.DataFrame(result['rows'], columns=result['header'])
            
            if len(result_df) > 0:
                return True, result_df, None
            else:
                return False, None, "SQL returned empty result"
    except TimeoutException as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def batch_execute_sql_with_retry(tablebench_data: List[Dict], 
                                 tablebench_df_processed: Dict[int, pd.DataFrame],
                                 executor: Executor, db: NeuralDB,
                                 args, template_path: str) -> List[Dict]:
    """
    Batch execute SQL with iterative retry logic and timeout.
    Uses infer_prompts for batch queries instead of one-by-one.
    Returns execution statistics for all samples.
    """
    # Initialize stats for all samples
    all_stats = []
    timeout_count = 0
    
    for idx, item in enumerate(tablebench_data):
        all_stats.append({
            'idx': idx,
            'question': item.get('question', ''),
            'iterations': [],
            'final_success': False,
            'final_result': None,
            'total_attempts': 0,
            'successful_attempts': 0,
            'timeout_count': 0
        })
    
    # Track which samples need processing in each iteration
    active_samples = set(range(len(tablebench_data)))
    
    for iteration in range(args.max_iterations):
        if not active_samples:
            break
        
        print(f"\n  Iteration {iteration}: Processing {len(active_samples)} samples...")
        
        # Build prompts for all active samples
        prompt_list = []
        idx_mapping = []  # Map prompt index to sample index
        
        for idx in sorted(active_samples):
            item = tablebench_data[idx]
            df = tablebench_df_processed[idx]
            stats = all_stats[idx]
            
            # Build prompt with error feedback if retry
            error_msg = None
            if iteration > 0 and stats['iterations']:
                last_iter = stats['iterations'][-1]
                if last_iter['attempts']:
                    # Get first error from last iteration
                    error_msg = last_iter['attempts'][0].get('error', 'SQL execution failed')
            
            prompt = build_sql_prompt(item, df, template_path, error_msg, iteration)
            prompt_list.append(prompt)
            idx_mapping.append(idx)
        
        # Batch generate SQL queries for all active samples
        print(f"  Generating SQL queries for {len(prompt_list)} samples...")
        responses, _, _ = infer_prompts(
            prompt_list,
            sample_num=args.sql_sample_num,
            temperature=args.temperature,
            top_p=args.top_p,
            llm_name=args.llm_name,
            api_base=args.api_base,
            api_key=args.api_key,
            concurrency=args.llm_concurrency
        )
        
        # Process responses for each sample
        newly_succeeded = set()
        
        for prompt_idx, response_list in enumerate(responses):
            idx = idx_mapping[prompt_idx]
            item = tablebench_data[idx]
            df = tablebench_df_processed[idx]
            stats = all_stats[idx]
            table_title = item.get('id', f'Table_{idx}')
            
            iteration_stats = {
                'iteration': iteration,
                'attempts': [],
                'success': False
            }
            
            # Try each generated SQL
            for sample_idx, response_text in enumerate(response_list):
                stats['total_attempts'] += 1
                attempt = {
                    'sample_idx': sample_idx,
                    'response': response_text,
                    'sql': None,
                    'success': False,
                    'error': None,
                    'result': None,
                    'timeout': False
                }
                
                # Parse SQL
                sql = fix_sql_query(
                    response_text=response_text,
                    table_df=df,
                    table_title=table_title
                )
                attempt['sql'] = sql
                
                if sql:
                    # Execute SQL with timeout
                    success, result_df, error_msg = execute_sql_with_timeout(
                        executor, sql, db, idx, args.sql_timeout
                    )
                    
                    if success:
                        attempt['success'] = True
                        attempt['result'] = result_df
                        iteration_stats['success'] = True
                        stats['successful_attempts'] += 1
                        stats['final_success'] = True
                        stats['final_result'] = result_df
                    else:
                        attempt['error'] = error_msg
                        if 'timed out' in error_msg.lower():
                            attempt['timeout'] = True
                            stats['timeout_count'] += 1
                            timeout_count += 1
                else:
                    attempt['error'] = "Failed to parse SQL from response"
                
                iteration_stats['attempts'].append(attempt)
            
            stats['iterations'].append(iteration_stats)
            
            # Mark as succeeded if any attempt worked
            if iteration_stats['success']:
                newly_succeeded.add(idx)
        
        # Remove succeeded samples from active set
        active_samples -= newly_succeeded
        print(f"  Iteration {iteration}: {len(newly_succeeded)} samples succeeded, {len(active_samples)} remaining")
    
    print(f"\n  Total SQL timeouts: {timeout_count}")
    return all_stats


def main():
    args = parse_args()
    
    # Add timestamp to save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.tmp_save_path = f"{args.tmp_save_path}_{timestamp}"
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    print("="*80)
    print("Simplified SQL Pipeline with Iterative Retry (Batch Version)")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Save Path: {args.tmp_save_path}")
    print(f"Dataset: {args.tablebench_jsonl_path}")
    print(f"Sample Size: {args.first_n}")
    print(f"Random Sample: {args.random_sample}")
    print(f"SQL Samples per Question: {args.sql_sample_num}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"SQL Timeout: {args.sql_timeout}s")
    print(f"LLM Name: {args.llm_name}")
    print(f"API Base: {args.api_base}")
    print(f"Temperature: {args.temperature}")
    print(f"Top P: {args.top_p}")
    print(f"LLM Concurrency: {args.llm_concurrency}")
    print("="*80)
    print()
    
    overall_start = time.perf_counter()
    
    # Load data
    print("[Step 1] Loading Data...")
    tablebench_data = load_tablebench_dataset(
        args.tablebench_jsonl_path, 
        args.first_n, 
        args.random_sample
    )
    
    # Preprocess tables
    print("\n[Step 2] Preprocessing Tables...")
    tablebench_df_processed = {}
    for idx, item in enumerate(tqdm(tablebench_data)):
        tablebench_df_processed[idx] = tablebench_table_to_df(item)
    
    # Build database
    print("\n[Step 3] Building Database...")
    table_titles = [tablebench_data[i].get('id', f'Table_{i}') for i in range(len(tablebench_data))]
    tables_for_db = [tablebench_df_processed[i] for i in range(len(tablebench_data))]
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    
    # Execute SQL with iterative retry (batch processing)
    print("\n[Step 4] Executing SQL with Iterative Retry (Batch)...")
    template_path = os.path.join(os.path.dirname(__file__), '../prompts/sql_reason_wtq.txt')
    
    all_stats = batch_execute_sql_with_retry(
        tablebench_data, 
        tablebench_df_processed, 
        executor, 
        db, 
        args, 
        template_path
    )
    
    # Calculate statistics
    print("\n[Step 5] Calculating Statistics...")
    total_samples = len(all_stats)
    total_attempts = sum(s['total_attempts'] for s in all_stats)
    successful_attempts = sum(s['successful_attempts'] for s in all_stats)
    final_success_count = sum(1 for s in all_stats if s['final_success'])
    total_timeouts = sum(s['timeout_count'] for s in all_stats)
    
    # Iteration statistics
    samples_success_iter_0 = sum(1 for s in all_stats if s['iterations'] and s['iterations'][0]['success'])
    samples_success_iter_1 = sum(1 for s in all_stats if len(s['iterations']) > 1 and s['iterations'][1]['success'] and not s['iterations'][0]['success'])
    samples_success_iter_2 = sum(1 for s in all_stats if len(s['iterations']) > 2 and s['iterations'][2]['success'] and not any(s['iterations'][i]['success'] for i in range(2)))
    
    # Save detailed stats
    stats_file = os.path.join(args.tmp_save_path, 'execution_stats_detailed.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'total_samples': total_samples,
        'total_attempts': total_attempts,
        'successful_attempts': successful_attempts,
        'final_success_count': final_success_count,
        'total_timeouts': total_timeouts,
        'sql_execution_success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
        'sample_success_rate': final_success_count / total_samples if total_samples > 0 else 0,
        'timeout_rate': total_timeouts / total_attempts if total_attempts > 0 else 0,
        'samples_success_iteration_0': samples_success_iter_0,
        'samples_success_iteration_1': samples_success_iter_1,
        'samples_success_iteration_2': samples_success_iter_2,
        'avg_attempts_per_sample': total_attempts / total_samples if total_samples > 0 else 0,
        'config': {
            'sql_sample_num': args.sql_sample_num,
            'max_iterations': args.max_iterations,
            'sql_timeout': args.sql_timeout,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'llm_concurrency': args.llm_concurrency
        }
    }
    
    summary_file = os.path.join(args.tmp_save_path, 'execution_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate final QA (batch processing)
    print("\n[Step 6] Generating Final QA (Batch)...")
    qa_template_path = os.path.join(os.path.dirname(__file__), '../prompts/text_reason_wtq_nocase.txt')
    
    with open(qa_template_path, 'r', encoding='utf-8') as f:
        qa_template = f.read()
    
    prompt_list = []
    for idx, stats in enumerate(all_stats):
        item = tablebench_data[idx]
        df = tablebench_df_processed[idx]
        
        # Build base QA prompt
        temp_df = df.copy()
        if 'row_id' not in temp_df.columns:
            temp_df.insert(0, 'row_id', temp_df.index)
        table_str = temp_df.to_string(index=False)
        
        columns = df.columns.tolist()
        all_cols = ['row_id'] + columns if 'row_id' not in columns else columns
        
        table_title = item.get('id', 'Table')
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(table_title))
        
        schema_cols = [f"`{col}` text" for col in columns]
        if 'row_id' not in columns:
            schema_cols.insert(0, "row_id int")
        col_defs = ",\n\t".join(schema_cols)
        schema = f"CREATE TABLE {table_name}(\n\t{col_defs})"
        
        question = item.get('question', '')
        
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
        
        prompt = qa_template.strip() + "\n" + input_section
        
        # Add SQL evidence if available (using table_to_str_sql which includes full format)
        if stats['final_success'] and stats['final_result'] is not None:
            evidence = table_to_str_sql(stats['final_result'])
            prompt = prompt + evidence
        
        prompt_list.append(prompt)
    
    # Batch generate QA responses
    print(f"  Generating QA responses for {len(prompt_list)} samples...")
    qa_responses, _, _ = infer_prompts(
        prompt_list,
        sample_num=1,
        temperature=0,
        top_p=1,
        llm_name=args.llm_name,
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.llm_concurrency
    )
    
    # Evaluate
    print("\n[Step 7] Evaluation...")
    
    # Extract predictions with proper string processing (same as run_full_pipeline_tablebench.py)
    preds = []
    for qa in qa_responses:
        pred_str = qa[0] if isinstance(qa, list) else str(qa)
        # Extract "The answer is:" pattern
        match = re.search(r'(?:the answer is|therefore|answer):\s*(.+)', pred_str, re.IGNORECASE)
        if match:
            pred_str = match.group(1).strip()
        pred_str = pred_str.strip().strip('"\'')[:200]  # Limit length to 200 characters
        preds.append(pred_str)
    
    golds = [str(item.get('answer', '')) for item in tablebench_data]
    
    eval_results = evaluate_tablebench_predictions(preds, golds)
    
    # Save results
    results_df = pd.DataFrame({
        'id': [item.get('id', idx) for idx, item in enumerate(tablebench_data)],
        'question': [item.get('question', '') for item in tablebench_data],
        'gold_answer': golds,
        'prediction': preds,
        'sql_success': [s['final_success'] for s in all_stats],
        'total_attempts': [s['total_attempts'] for s in all_stats],
        'successful_attempts': [s['successful_attempts'] for s in all_stats],
        'timeout_count': [s['timeout_count'] for s in all_stats],
        'iterations_used': [len(s['iterations']) for s in all_stats]
    })
    results_df.to_csv(os.path.join(args.tmp_save_path, 'results.csv'), index=False)
    
    total_time = time.perf_counter() - overall_start
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Samples: {total_samples}")
    print(f"Total SQL Attempts: {total_attempts}")
    print(f"Successful SQL Executions: {successful_attempts}")
    print(f"SQL Timeouts: {total_timeouts}")
    print(f"SQL Execution Success Rate: {summary['sql_execution_success_rate']*100:.2f}%")
    print(f"SQL Timeout Rate: {summary['timeout_rate']*100:.2f}%")
    print(f"Sample Success Rate: {summary['sample_success_rate']*100:.2f}%")
    print()
    print("Success by Iteration:")
    print(f"  Iteration 0 (initial): {samples_success_iter_0} samples")
    print(f"  Iteration 1 (retry 1): {samples_success_iter_1} samples")
    print(f"  Iteration 2 (retry 2): {samples_success_iter_2} samples")
    print(f"  Total Success: {final_success_count} samples")
    print()
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Average ROUGE-L: {eval_results['avg_rouge_l']:.4f}")
    print(f"Accuracy@0.5: {eval_results['accuracy_at_0.5']*100:.2f}%")
    print(f"Accuracy@0.8: {eval_results['accuracy_at_0.8']*100:.2f}%")
    print()
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {args.tmp_save_path}")
    print("="*80)


if __name__ == "__main__":
    main()
