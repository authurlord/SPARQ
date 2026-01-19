#!/usr/bin/env python3
"""
TableBench PoT (Program-of-Thought) Pipeline
Solved via Python Code Execution

Steps:
1. Data Loading & Preprocessing
2. Generate Python Code (n=3 samples)
3. Execute Python Code & Select Result
4. Generate Final Answer (using execution result as context)
5. Evaluate (ROUGE-L)
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List
import re
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_llm import infer_prompts
from utils.schedule_utils import table_to_str_sql
from utils.evaluator import evaluate_tablebench_predictions
from utils.python_executor import execute_python_code

import multiprocessing as mp

# Multiprocessing setup
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="TableBench PoT Pipeline")
    
    parser.add_argument('--llm_path', type=str, 
                       default='/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/', help='Path to LLM model')
    parser.add_argument('--dataset_name', type=str, default='tablebench', help='Dataset name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/tablebench_pot',
                       help='Temporary save path')
    parser.add_argument('--tablebench_jsonl_path', type=str,
                       default='../datasets/TableBench/TableBench_PoT.jsonl',
                       help='Path to TableBench PoT JSONL file')
    
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--llm_concurrency', type=int, default=32, help='Max concurrent requests')
    
    # Sampling parameters for Code Generation
    parser.add_argument('--code_sample_num', type=int, default=3, help='Samples for Python code generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8, help='Sampling top_p')
    
    parser.add_argument('--first_n', type=int, default=-1, help='Only process first N samples')
    parser.add_argument('--use_api', action='store_true', help='Use async API')
    
    # API Arguments (Added repair)
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1", help='vLLM API Base URL')
    parser.add_argument('--api_key', type=str, default="EMPTY", help='vLLM API Key')
    
    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================

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
    """Builds the PoT prompt using JSON table representation."""
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
    """Builds final QA prompt with Python execution result as context."""
    # Load template
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = "" # Fallback?

    temp_df = df.copy()
    if 'row_id' not in temp_df.columns:
        temp_df.insert(0, 'row_id', temp_df.index)
    table_str = temp_df.to_string(index=False)
    
    columns = df.columns.tolist()
    all_cols = ['row_id'] + columns
    
    # Clean table title
    table_title = item.get('id', 'Table')
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(table_title))
    
    # Pseudo-schema
    schema_cols = [f"`{col}` text" for col in columns]
    if 'row_id' not in columns:
        schema_cols.insert(0, "row_id int")
    col_defs = ",\n\t".join(schema_cols)
    schema = f"CREATE TABLE {table_name}(\n\t{col_defs})"
    
    question = item.get('question', '')
    
    # Construct Execution Evidence
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
    """Extracts code block from response."""
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1)
    
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1)
        
    return response

# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    overall_start = time.perf_counter()
    timeline = {}
    metrics = {}
    
    # 1. Load Data
    print("Loading Data...")
    raw_data = load_tablebench_dataset(args.tablebench_jsonl_path, args.first_n)
    
    # Preprocess DataFrames
    print("Preprocessing tables...")
    processed_dfs = {}
    for idx, item in enumerate(tqdm(raw_data)):
        processed_dfs[idx] = tablebench_table_to_df(item)
        
    # 2. Generate Python Code
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
        llm_name=args.llm_path, # Pass path as model name for vLLM
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.llm_concurrency
    )
    
    timeline['Generate Python'] = time.perf_counter() - _t1
    
    # 3. Execute Python Code
    print("\n[Step 2] Executing Python Code...")
    _t2 = time.perf_counter()
    
    execution_results = {} # idx -> best_result_string
    error_log_path = os.path.join(args.tmp_save_path, "execution_errors.log")
    
    with open(error_log_path, 'w', encoding='utf-8') as error_log:
        for idx in tqdm(range(len(raw_data)), desc="Executing"):
            responses = code_responses[idx]
            df = processed_dfs[idx]
            
            results = []
            for i, r in enumerate(responses):
                code = extract_python_code(r)
                output = execute_python_code(code, df)
                
                # Simple heuristic: filter out errors or empty outputs if possible
                if output and "Execution Error" not in output:
                    results.append(output.strip())
                else:
                     # Log failure
                    error_log.write(f"=== Error Sample {idx} (Code {i}) ===\n")
                    error_log.write(f"Question: {raw_data[idx].get('question', '')}\n")
                    error_log.write(f"Attempt: {i}\n")
                    error_log.write("Code:\n")
                    error_log.write(code + "\n")
                    error_log.write("Output/Error:\n")
                    error_log.write(str(output) + "\n\n")
            
            # Selection Strategy: Majority Voting or First Valid
            final_output = ""
            if results:
                # Majority vote
                counter = Counter(results)
                most_common = counter.most_common(1)[0]
                final_output = most_common[0]
            else:
                pass
                
            execution_results[idx] = final_output
        
    timeline['Execute Python'] = time.perf_counter() - _t2
    
    # 4. Generate Final Answer
    print("\n[Step 3] Generating Final QA...")
    _t3 = time.perf_counter()
    
    qa_prompt_list = []
    qa_template_path = os.path.join(os.path.dirname(__file__), '../prompts/text_reason_wtq.txt')
    
    for idx, item in enumerate(raw_data):
        res = execution_results.get(idx, "")
        prompt = build_final_qa_prompt(item, processed_dfs[idx], res, qa_template_path)
        qa_prompt_list.append(prompt)
        
    qa_responses, metrics['final_qa'], _ = infer_prompts(
        qa_prompt_list,
        sample_num=1, # Greedy generation for final answer
        temperature=0,
        top_p=1,
        llm_name=args.llm_path, # Pass path as model name
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.llm_concurrency
    )
    
    timeline['Final QA'] = time.perf_counter() - _t3
    
    # 5. Evaluate
    print("\n[Step 4] Evaluation...")
    
    preds = []
    golds = []
    
    output_data = [] # For CSV saving
    
    for idx, item in enumerate(raw_data):
        pred = qa_responses[idx][0] if isinstance(qa_responses[idx], list) else qa_responses[idx]
        gold = item.get('answer', '')
        
        preds.append(str(pred))
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
    
    # Save results
    res_df = pd.DataFrame(output_data)
    res_df.to_csv(f"{args.tmp_save_path}/pot_results.csv", index=False)
    
    with open(f"{args.tmp_save_path}/evaluation.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
        
    total_time = time.perf_counter() - overall_start
    print(f"Total Time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
