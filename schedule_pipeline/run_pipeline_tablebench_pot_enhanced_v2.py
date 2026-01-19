#!/usr/bin/env python3
"""
Enhanced TableBench PoT Pipeline with detailed logging
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List
import re
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_llm import infer_prompts
from utils.schedule_utils import table_to_str_sql
from utils.evaluator import evaluate_tablebench_predictions
from utils.python_executor import execute_python_code

import multiprocessing as mp

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced TableBench PoT Pipeline")
    
    parser.add_argument('--llm_path', type=str, 
                       default='/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/', help='Path to LLM model')
    parser.add_argument('--llm_name', type=str, 
                       default='qwen3-4b', help='Model name registered in vLLM API server')
    parser.add_argument('--dataset_name', type=str, default='tablebench', help='Dataset name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/tablebench_pot_enhanced',
                       help='Temporary save path')
    parser.add_argument('--tablebench_jsonl_path', type=str,
                       default='../datasets/TableBench/TableBench_PoT.jsonl',
                       help='Path to TableBench PoT JSONL file')
    
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--llm_concurrency', type=int, default=32, help='Max concurrent requests')
    
    parser.add_argument('--code_sample_num', type=int, default=3, help='Samples for Python code generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8, help='Sampling top_p')
    
    parser.add_argument('--first_n', type=int, default=-1, help='Only process first N samples')
    parser.add_argument('--use_api', action='store_true', help='Use async API')
    
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1", help='vLLM API Base URL')
    parser.add_argument('--api_key', type=str, default="api-key-qwen3", help='vLLM API Key')
    
    return parser.parse_args()

def load_tablebench_dataset(jsonl_path: str, first_n: int = -1) -> List[Dict]:
    print(f"Loading TableBench dataset from {jsonl_path}...")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
                if first_n > 0 and len(data) >= first_n:
                    break
    print(f"Loaded {len(data)} samples")
    return data

def make_unique_columns(columns: List[str]) -> List[str]:
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
    columns = item['table']['columns']
    data = item['table']['data']
    unique_columns = make_unique_columns(columns)
    return pd.DataFrame(data, columns=unique_columns)

def build_tablebench_pot_prompt(item: Dict, df: pd.DataFrame, template_path: str) -> str:
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    table_dict = {
        'columns': df.columns.tolist(),
        'data': df.values.tolist()
    }
    table_str = str(table_dict) 
    
    question = item.get('question', '')
    
    prompt = template.strip() + f"\n\nRead the table below in JSON format:\n[TABLE] \n{table_str}\n\nLet's get start!\nQuestion: {question}"
    return prompt

def build_final_qa_prompt(item: Dict, df: pd.DataFrame, execution_result: str, 
                          template_path: str = '../prompts/text_reason_wtq.txt') -> str:
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = ""

    temp_df = df.copy()
    if 'row_id' not in temp_df.columns:
        temp_df.insert(0, 'row_id', temp_df.index)
    table_str = temp_df.to_string(index=False)
    
    columns = df.columns.tolist()
    all_cols = ['row_id'] + columns
    
    table_title = item.get('id', 'Table')
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(table_title))
    
    schema_cols = [f"`{col}` text" for col in columns]
    if 'row_id' not in columns:
        schema_cols.insert(0, "row_id int")
    col_defs = ",\n\t".join(schema_cols)
    schema = f"CREATE TABLE {table_name}(\n\t{col_defs})"
    
    question = item.get('question', '')
    
    evidence = ""
    if execution_result:
        evidence = f"\nHere is an additional evidence to help the answering process.\nAdditional Evidence:\n/*\nPython Output:\n{execution_result}\n*/\n"
    
    input_section = f"""
<input>
{schema}
/*
SELECT * FROM w;
{table_str}
*/
columns: {all_cols}
Q: {question}
{evidence}
<output>"""

    if template:
        prompt = template.strip() + "\n" + input_section
    else:
        prompt = f"Table:\n{table_str}\n\nEvidence: {execution_result}\n\nQuestion: {question}"
        
    return prompt

def extract_python_code(response: str) -> str:
    """Enhanced code extraction with better parsing logic."""
    # Try to find code block with python tag
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find any code block
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like Python code
        if any(keyword in code for keyword in ['import', 'def ', 'print', 'pd.', 'df']):
            return code
    
    # Try to find code after "Step" markers
    if 'Step' in response and '```' not in response:
        lines_resp = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines_resp:
            if line.strip().startswith('import ') or line.strip().startswith('df '):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
    
    # Check if response contains Python-like code without markers
    if any(keyword in response for keyword in ['import pandas', 'pd.read_csv', 'df =', 'print(']):
        lines_resp = response.split('\n')
        code_lines = []
        for line in lines_resp:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('Step'):
                if any(kw in stripped for kw in ['import', 'pd.', 'df', 'print', '=', 'mean(', 'sum(']):
                    code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
    
    return ""


def main():
    args = parse_args()
    
    # Add timestamp to save path to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tmp_save_path.endswith('_enhanced'):
        args.tmp_save_path = f"{args.tmp_save_path}_{timestamp}"
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    # Print all key parameters at the start
    print("="*80)
    print("TableBench PoT Pipeline - Enhanced Version")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Save Path: {args.tmp_save_path}")
    print(f"Dataset: {args.tablebench_jsonl_path}")
    print(f"First N: {args.first_n}")
    print(f"LLM Name: {args.llm_name}")
    print(f"API Base: {args.api_base}")
    print(f"Code Sample Num: {args.code_sample_num}")
    print(f"Temperature: {args.temperature}")
    print(f"Top P: {args.top_p}")
    print(f"Concurrency: {args.llm_concurrency}")
    print("="*80)
    print()
    
    overall_start = time.perf_counter()
    timeline = {}
    metrics = {}
    
    print("Loading Data...")
    raw_data = load_tablebench_dataset(args.tablebench_jsonl_path, args.first_n)
    
    print("Preprocessing tables...")
    processed_dfs = {}
    for idx, item in enumerate(tqdm(raw_data)):
        processed_dfs[idx] = tablebench_table_to_df(item)
        
    print("\n[Step 1] Generating Python Code...")
    _t1 = time.perf_counter()
    
    prompt_list = []
    template_path = os.path.join(os.path.dirname(__file__), '../prompts/python_reason_tablebench.txt')
    
    for idx, item in enumerate(raw_data):
        prompt = build_tablebench_pot_prompt(item, processed_dfs[idx], template_path)
        prompt_list.append(prompt)
        
    code_responses, metrics['code_gen'], _ = infer_prompts(
        prompt_list,
        sample_num=args.code_sample_num,
        temperature=args.temperature,
        top_p=args.top_p,
        llm_name=args.llm_name,
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.llm_concurrency
    )
    
    timeline['Generate Python'] = time.perf_counter() - _t1
    
    print("\n[Step 2] Executing Python Code...")
    _t2 = time.perf_counter()
    
    execution_results = {}
    execution_stats = {
        'total_samples': len(raw_data),
        'total_attempts': 0,
        'successful_executions': 0,
        'failed_executions': 0,
        'parse_failures': 0,
        'samples_with_all_failures': 0,
        'samples_with_partial_success': 0,
        'samples_with_all_success': 0
    }
    
    code_dir = os.path.join(args.tmp_save_path, "generated_codes")
    os.makedirs(code_dir, exist_ok=True)
    
    error_log_path = os.path.join(args.tmp_save_path, "execution_errors.log")
    detailed_log_path = os.path.join(args.tmp_save_path, "execution_detailed.log")
    
    with open(error_log_path, 'w', encoding='utf-8') as error_log, \
         open(detailed_log_path, 'w', encoding='utf-8') as detailed_log:
        
        detailed_log.write("="*80 + "\n")
        detailed_log.write("DETAILED EXECUTION LOG\n")
        detailed_log.write("="*80 + "\n\n")
        
        for idx in tqdm(range(len(raw_data)), desc="Executing"):
            responses = code_responses[idx]
            df = processed_dfs[idx]
            question = raw_data[idx].get('question', '')
            sample_id = raw_data[idx].get('id', idx)
            
            detailed_log.write(f"\n{'='*80}\n")
            detailed_log.write(f"Sample {idx} (ID: {sample_id})\n")
            detailed_log.write(f"Question: {question}\n")
            detailed_log.write(f"{'='*80}\n")
            
            results = []
            success_count = 0
            
            for i, r in enumerate(responses):
                execution_stats['total_attempts'] += 1
                
                response_file = os.path.join(code_dir, f"sample_{idx}_attempt_{i}_response.txt")
                with open(response_file, 'w', encoding='utf-8') as f:
                    f.write(r)
                
                code = extract_python_code(r)
                
                if not code or len(code.strip()) < 10:
                    execution_stats['parse_failures'] += 1
                    detailed_log.write(f"\n--- Attempt {i}: PARSE FAILURE ---\n")
                    detailed_log.write(f"Response length: {len(r)}\n")
                    detailed_log.write(f"Preview: {r[:200]}...\n")
                    
                    error_log.write(f"=== Parse Failure: Sample {idx} Attempt {i} ===\n")
                    error_log.write(f"Question: {question}\n")
                    error_log.write(f"Response:\n{r}\n\n")
                    continue
                
                code_file = os.path.join(code_dir, f"sample_{idx}_attempt_{i}.py")
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                output = execute_python_code(code, df)
                
                if output and "Execution Error" not in output:
                    results.append(output.strip())
                    success_count += 1
                    execution_stats['successful_executions'] += 1
                    
                    detailed_log.write(f"\n--- Attempt {i}: SUCCESS ---\n")
                    detailed_log.write(f"Code: {code_file}\n")
                    detailed_log.write(f"Output: {output.strip()}\n")
                else:
                    execution_stats['failed_executions'] += 1
                    
                    detailed_log.write(f"\n--- Attempt {i}: EXECUTION FAILURE ---\n")
                    detailed_log.write(f"Code: {code_file}\n")
                    detailed_log.write(f"Error: {output}\n")
                    
                    error_log.write(f"=== Execution Error: Sample {idx} Attempt {i} ===\n")
                    error_log.write(f"Question: {question}\n")
                    error_log.write(f"Code file: {code_file}\n")
                    error_log.write(f"Code:\n{code}\n")
                    error_log.write(f"Error:\n{output}\n\n")
            
            if success_count == 0:
                execution_stats['samples_with_all_failures'] += 1
            elif success_count == len(responses):
                execution_stats['samples_with_all_success'] += 1
            else:
                execution_stats['samples_with_partial_success'] += 1
            
            final_output = ""
            if results:
                counter = Counter(results)
                most_common = counter.most_common(1)[0]
                final_output = most_common[0]
                detailed_log.write(f"\nSelected: {final_output}\n")
            else:
                detailed_log.write(f"\nNo successful execution\n")
                
            execution_results[idx] = final_output
    
    stats_path = os.path.join(args.tmp_save_path, "execution_stats.json")
    execution_stats['success_rate'] = execution_stats['successful_executions'] / execution_stats['total_attempts'] if execution_stats['total_attempts'] > 0 else 0
    execution_stats['parse_failure_rate'] = execution_stats['parse_failures'] / execution_stats['total_attempts'] if execution_stats['total_attempts'] > 0 else 0
    
    with open(stats_path, 'w') as f:
        json.dump(execution_stats, f, indent=2)
    
    print(f"\n--- Execution Statistics ---")
    print(f"Total samples: {execution_stats['total_samples']}")
    print(f"Total attempts: {execution_stats['total_attempts']}")
    print(f"Successful: {execution_stats['successful_executions']} ({execution_stats['success_rate']*100:.1f}%)")
    print(f"Failed: {execution_stats['failed_executions']}")
    print(f"Parse failures: {execution_stats['parse_failures']} ({execution_stats['parse_failure_rate']*100:.1f}%)")
    print(f"All success: {execution_stats['samples_with_all_success']}")
    print(f"Partial success: {execution_stats['samples_with_partial_success']}")
    print(f"All failures: {execution_stats['samples_with_all_failures']}")
    print(f"----------------------------\n")
        
    timeline['Execute Python'] = time.perf_counter() - _t2
    
    print("\n[Step 3] Generating Final QA...")
    _t3 = time.perf_counter()
    
    qa_prompt_list = []
    qa_template_path = os.path.join(os.path.dirname(__file__), '../prompts/text_reason_wtq_nocase.txt')
    
    for idx, item in enumerate(raw_data):
        res = execution_results.get(idx, "")
        prompt = build_final_qa_prompt(item, processed_dfs[idx], res, qa_template_path)
        qa_prompt_list.append(prompt)
        
    qa_responses, metrics['final_qa'], _ = infer_prompts(
        qa_prompt_list,
        sample_num=1,
        temperature=0,
        top_p=1,
        llm_name=args.llm_name,
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.llm_concurrency
    )
    
    timeline['Final QA'] = time.perf_counter() - _t3
    
    print("\n[Step 4] Evaluation...")
    
    preds = []
    golds = []
    
    output_data = []
    
    for idx, item in enumerate(raw_data):
        pred = qa_responses[idx][0] if isinstance(qa_responses[idx], list) else qa_responses[idx]
        gold = item.get('answer', '')
        
        # Extract predictions with proper string processing

        
        pred_str = str(pred)

        
        # Extract "The answer is:" pattern

        
        match = re.search(r'(?:the answer is|therefore|answer):\s*(.+)', pred_str, re.IGNORECASE)

        
        if match:

        
            pred_str = match.group(1).strip()

        
        pred_str = pred_str.strip().strip('"\'')[:200]  # Limit length to 200 characters

        
        preds.append(pred_str)
        golds.append(str(gold))
        
        output_data.append({
            'id': item.get('id', idx),
            'question': item.get('question', ''),
            'code_generated': code_responses[idx][0] if code_responses[idx] else "",
            'execution_result': execution_results.get(idx, ""),
            'final_prompt': qa_prompt_list[idx],
            'prediction': str(pred),
            'gold': str(gold)
        })
        
    eval_results = evaluate_tablebench_predictions(preds, golds)
    
    print("="*40)
    print(f"ROUGE-L: {eval_results['avg_rouge_l']:.4f}")
    print(f"Acc@0.5: {eval_results['accuracy_at_0.5']:.4f}")
    print("="*40)
    
    res_df = pd.DataFrame(output_data)
    res_df.to_csv(f"{args.tmp_save_path}/pot_results.csv", index=False)
    
    with open(f"{args.tmp_save_path}/evaluation.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
        
    total_time = time.perf_counter() - overall_start
    print(f"Total Time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
