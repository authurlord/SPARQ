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
from utils.python_executor import execute_python_code  # Newly created

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
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1", help='vLLM API Base URL')
    parser.add_argument('--api_key', type=str, default="EMPTY", help='vLLM API Key')
    
    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================
# ... (no changes to helpers) ...

def build_final_qa_prompt(item: Dict, df: pd.DataFrame, execution_result: str, 
                          template_path: str = '../prompts/text_reason_wtq.txt') -> str:
# ... (ensure content matches) ...
    # Reuse existing prompt logic but inject evidence
    
    # Load template
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = "" # Fallback?

    # Format like NIAT/WTQ: <input> ...
    # We need to construct the SQL-like table representation because text_reason_wtq expect it?
    # Or matches what we did in run_full_pipeline.
    
    # Let's trust run_full_pipeline's build_tablebench_prompt_from_df mostly, 
    # but we need to append the evidence manually since we don't have SQL structure.
    
    # Format table for context
    # Reuse logic from run_full_pipeline logic (simple str representation)
    
    # We will construct a minimal prompt here to avoid import circularity or complexity
    # Actually, let's copy the logic from table_to_str_tablebench in run_full.
    
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

# ...

# Inside main() ...

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
    
# ...

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
