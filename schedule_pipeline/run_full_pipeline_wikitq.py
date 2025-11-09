#!/usr/bin/env python3
"""
Complete H-STAR Pipeline for WikiTQ
Integrates all steps from the notebook into a single executable script
Follows the exact logic from schedule_pipeline_wikitq.ipynb
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
from schedule_pipeline.gpu_monitor import GPUMonitor

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

# Ensure vLLM uses spawn for CUDA-safe multiprocessing
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# 限制 OpenMP 线程数，避免 FlagEmbedding 多进程冲突
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
    parser = argparse.ArgumentParser(description="H-STAR Full Pipeline for WikiTQ")
    
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
    parser.add_argument('--dataset_name', type=str, default='wikitq',
                       choices=['wikitq', 'tab_fact'],
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/wikitq',
                       help='Temporary save path for intermediate results')
    
    # Pipeline parameters
    parser.add_argument('--tau', type=float, default=0.82,
                       help='Router threshold')
    parser.add_argument('--check_tau', type=float, default=0.8,
                       help='Check model threshold')
    parser.add_argument('--n_parallel', type=int, default=32,
                       help='Number of parallel workers for preprocessing')
    
    # vLLM parameters
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
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
                       help='Max concurrent requests to vLLM API (lower to reduce VRAM)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8,
                       help='Sampling top_p')
    
    # Execution control
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--skip_router', action='store_true',
                       help='Skip router inference if already done')
    parser.add_argument('--skip_rag', action='store_true',
                       help='Skip RAG if already done')
    parser.add_argument('--first_n', type=int, default=-1,
                       help='Only process first N samples (-1 for all)')
    
    return parser.parse_args()


def response_vllm(llm, tokenizer, all_instructions: List[str], sample_num: int, 
                 temperature: float = 0.7, top_p: float = 0.8) -> List[List[str]]:
    """
    Generate responses using vLLM - exactly as in notebook Cell 5
    """
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
    """Initialize vLLM and tokenizer - exactly as in notebook Cell 4"""
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


def main():
    args = parse_args()
    
    print("="*80)
    print("H-STAR Full Pipeline for WikiTQ")
    print("="*80)
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    ALL_LABELS = ['Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5']
    
    # Initialize vLLM at the beginning
    # print("\n[Init] Initializing vLLM...")
    # llm, tokenizer = init_llm_and_tokenizer(args)

    # Timeline tracking
    timeline = {}
    start_time = time.perf_counter()
    overall_start = time.perf_counter()
    
    # ========================================================================
    # Step 1: Data Preprocessing - Cell 10
    # ========================================================================
    print("\n[Step 1] Data Preprocessing...")
    _t1 = time.perf_counter()
    
    preprocess_file = f'{args.tmp_save_path}/wikitq_df_processed.npy'
    
    if not args.skip_preprocess and not os.path.exists(preprocess_file):
        cmd = f"python convert_df_type_parallel.py --dataset_name {args.dataset_name} " \
              f"--split {args.split} --output_path {preprocess_file} " \
              f"--num_workers {args.n_parallel}"
        print(f"Running: {cmd}")
        os.system(cmd)
    else:
        print(f"Skipping preprocessing (file exists or --skip_preprocess)")
    
    # Load preprocessed data - Cell 11
    dataset = load_data_split(args.dataset_name, args.split)
    print(f"Dataset type: {type(dataset)}")

    if args.first_n > 0:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))
        print(f"Processing only first {args.first_n} samples")
    
    wikitq_df_processed = np.load(preprocess_file, allow_pickle=True).item()
    print(f"Loaded {len(dataset)} samples")
    
    timeline['Step 1 - Data Preprocessing'] = time.perf_counter() - _t1
    print(f"  [Timing] Step 1 - Data Preprocessing: {timeline['Step 1 - Data Preprocessing']:.2f}s")

    ## start up
    mon = GPUMonitor(interval_sec=1.0)
    mon.start()

    # ========================================================================
    # Step 2: Build Router Query File - Cell 12
    # ========================================================================
    print("\n[Step 2] Building Router Query File...")
    _t2 = time.perf_counter()
    
    semantic_router = {}
    for index in range(len(dataset)):
        semantic_router[index] = {}
        semantic_router[index]['query'] = dataset[index]['question']
        semantic_router[index]['title'] = dataset[index]['table']['page_title']
        semantic_router[index]['table'] = wikitq_df_processed[index]
        label_list = []
        semantic_router[index]['label'] = label_list
    
    with open(f'{args.tmp_save_path}/router_query.pkl', 'wb') as f:
        pickle.dump(semantic_router, f)
    print(f"Saved router query to {args.tmp_save_path}/router_query.pkl")
    
    timeline['Step 2 - Build Router Query'] = time.perf_counter() - _t2
    print(f"  [Timing] Step 2 - Build Router Query: {timeline['Step 2 - Build Router Query']:.2f}s")
    
    # ========================================================================
    # Step 3: Construct Database - Cell 13
    # ========================================================================
    print("\n[Step 3] Constructing Database...")
    _t3 = time.perf_counter()
    
    table_titles = [dataset[i]['table']['page_title'] for i in range(len(dataset))]
    # Respect --first_n: only build DB over the current dataset slice
    tables_for_db = [wikitq_df_processed[i] for i in range(len(dataset))]
    db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
    executor = Executor()
    print("Database initialized")
    
    timeline['Step 3 - Construct Database'] = time.perf_counter() - _t3
    print(f"  [Timing] Step 3 - Construct Database: {timeline['Step 3 - Construct Database']:.2f}s")
    
    # ========================================================================
    # Step 4: Inference Router Model - Cell 14
    # ========================================================================
    print("\n[Step 4] Router Model Inference...")
    _t4 = time.perf_counter()
    
    router_result_file = f'{args.tmp_save_path}/inference_result.pkl'
    
    cmd = f"python inference_router.py --input_path {args.tmp_save_path}/router_query.pkl " \
            f"--model_path {args.router_model_path} " \
            f"--output_path {router_result_file}"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    # Load router results - Cell 15
    with open(router_result_file, 'rb') as f:
        error_analysis_row = pickle.load(f)
    
    timeline['Step 4 - Router Inference'] = time.perf_counter() - _t4
    print(f"  [Timing] Step 4 - Router Inference: {timeline['Step 4 - Router Inference']:.2f}s")
    
    # ========================================================================
    # Step 5: Parse Router Results & Organize LLM Query List - Cell 15, 17
    # ========================================================================
    print("\n[Step 5] Parsing Router Results...")
    _t5 = time.perf_counter()
    
    ranked_result = process_error_analysis_list(
        error_analysis_row, truncate=True, tau=args.tau
    )
    
    print("Router result distribution:")
    print(pd.DataFrame([str(r) for r in ranked_result.values()]).value_counts())
    
    # Organize LLM query list - Cell 17
    LLM_query_list = {}
    for method in ALL_LABELS:
        LLM_query_list[method] = {}
        LLM_query_list[method]['index'] = []
        LLM_query_list[method]['query'] = []
        LLM_query_list[method]['qa'] = []
    
    for index in range(len(dataset)):
        for method in ALL_LABELS:
            if method in ranked_result[index]:
                LLM_query_list[method]['index'].extend([index])
                if method == 'Select_Column':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[index], index,
                        template_path='../prompts/col_select_sql.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].extend([prompt])
                elif method == 'Select_Row':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[index], index,
                        template_path='../prompts/row_select_sql.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].extend([prompt])
                elif method == 'Execute_SQL':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[index], index,
                        template_path='../prompts/sql_reason_wtq.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].extend([prompt])
    
    # Save RAG index - Cell 18
    np.save(f'{args.tmp_save_path}/RAG_index.npy', LLM_query_list['RAG_20_5']['index'])
    
    print(f"Query counts:")
    for method in ALL_LABELS:
        print(f"  {method}: {len(LLM_query_list[method]['index'])}")
    
    timeline['Step 5 - Parse Router & Build Queries'] = time.perf_counter() - _t5
    print(f"  [Timing] Step 5 - Parse Router & Build Queries: {timeline['Step 5 - Parse Router & Build Queries']:.2f}s")
    
    # ========================================================================
    # Step 6: Execute RAG Task - Cell 20
    # ========================================================================
    rag_count = len(LLM_query_list['RAG_20_5']['index'])
    rag_output_file = f'{args.tmp_save_path}/Hybrid_Retrieve_output.npy'
    
    print(f"\n[Step 6] Executing RAG on {rag_count} samples...")
    _t6 = time.perf_counter()

    cmd = f"python Hybrid_Retrieve_Update_dict.py --model_path {args.embedding_model_path} " \
            f"--dataset_name {args.dataset_name} --split {args.split} " \
            f"--index_path {args.tmp_save_path}/RAG_index.npy " \
            f"--output_path {rag_output_file} " \
            f"--max_rows 50 --max_cols 10 " \
            f"--processed_df_path {args.tmp_save_path}/wikitq_df_processed.npy " \
            f"--rewrite_query_path datasets/schedule_test/{args.dataset_name}/rewrite_query.npy"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    RAG_20_5 = np.load(f'{args.tmp_save_path}/Hybrid_Retrieve_output.npy',allow_pickle=True).item()
    
    timeline['Step 6 - RAG'] = time.perf_counter() - _t6
    print(f"  [Timing] Step 6 - RAG: {timeline['Step 6 - RAG']:.2f}s")


    # ========================================================================
    # Step 7: Execute LLM Queries - Cell 22
    # ========================================================================
    print("\n[Step 7] Executing Select_Row and Select_Column...")
    _t7 = time.perf_counter()
    
    metrics_rows = {}
    summaries = {}
    for method in ['Select_Row', 'Select_Column']:
        if LLM_query_list[method]['query']:
            prompt_list = LLM_query_list[method]['query']
            # response_list = response_vllm(
            #     llm, tokenizer, prompt_list,
            #     sample_num=args.select_sample_num,
            #     temperature=args.temperature,
            #     top_p=args.top_p
            # )
            response_list,metrics_rows[method],summaries[method] = infer_prompts(
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
    print(f"  [Timing] Step 7 - Select Ops Generation: {timeline['Step 7 - Select Ops Generation']:.2f}s")
    
    print("\n[Step 8] Executing Execute_SQL...")
    _t8 = time.perf_counter()
    
    if LLM_query_list['Execute_SQL']['query']:
        prompt_list = LLM_query_list['Execute_SQL']['query']
        # response_list = response_vllm(
        #     llm, tokenizer, prompt_list,
        #     sample_num=args.sql_sample_num,
        #     temperature=args.temperature,
        #     top_p=args.top_p
        # )
        response_list,metrics_rows["Execute_SQL"],summaries["Execute_SQL"] = infer_prompts(
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
    print(f"  [Timing] Step 8 - SQL Generation: {timeline['Step 8 - SQL Generation']:.2f}s")
    
    # ========================================================================
    # Step 9: SQL Parse and Execute - Cells 24, 26, 28
    # ========================================================================
    print("\n[Step 9] Parsing and Executing SQL...")
    _t9 = time.perf_counter()
    
    # Parse Select_Row - Cell 24
    print("  Parsing Select_Row SQL...")
    sub_table_list_all = {}
    filtered_tables_row = {}
    row_sql_index_list = LLM_query_list['Select_Row']['index']
    row_sql_response_list = LLM_query_list['Select_Row']['response']
    
    for i in range(len(row_sql_index_list)):
        sample_num = [0, 1]
        sub_table_list = []
        for sample_index in sample_num:
            index = row_sql_index_list[i]
            original_text = row_sql_response_list[i][sample_index]
            sql = fix_sql_query(
                response_text=original_text,
                table_df=wikitq_df_processed[index],
                table_title=table_titles[index]
            )
            try:
                result_1 = executor.sql_exec(
                    sql.replace('``', '`').replace("COUNT(*)", "*"),
                    db, table_id=index
                )
                sub_table_list.append(
                    pd.DataFrame(result_1['rows'], columns=result_1['header'])
                )
            except:
                continue
        
        sub_table_list_all[index] = sub_table_list
        filtered_df = retrieve_rows_by_subtables(
            wikitq_df_processed[index], sub_table_list
        )
        if len(filtered_df) == 0:
            filtered_df = wikitq_df_processed[index]
        filtered_tables_row[index] = filtered_df
    
    # Parse Select_Column - Cell 26
    print("  Parsing Select_Column SQL...")
    filtered_tables = {}
    filtered_headers = {}
    col_sql_index_list = LLM_query_list['Select_Column']['index']
    col_sql_response_list = LLM_query_list['Select_Column']['response']
    
    for i in range(len(col_sql_index_list)):
        ind = col_sql_index_list[i]
        input_df = wikitq_df_processed[ind]
        response_list = col_sql_response_list[i]
        assert isinstance(response_list, list)
        filtered_table, final_headers = filter_dataframe_from_responses(
            response_list, input_df, add_row_id=True
        )
        filtered_tables[ind] = filtered_table
        filtered_headers[ind] = final_headers
    
    # Parse Execute_SQL - Cell 28
    print("  Parsing Execute_SQL...")
    sample_num = 3
    sql_exec_df = {}
    valid_parse = 0
    warning_list = []
    sql_executable_count = []
    exec_sql_index_list = LLM_query_list['Execute_SQL']['index']
    exec_sql_response_list = LLM_query_list['Execute_SQL']['response']
    
    for i in range(len(exec_sql_index_list)):
        index = exec_sql_index_list[i]
        sql_exec_df[index] = []
        for sample_ind in range(sample_num):
            original_text = exec_sql_response_list[i][sample_ind]
            sql = fix_sql_query(
                response_text=original_text,
                table_df=wikitq_df_processed[index],
                table_title=table_titles[index]
            )
            if [index, sample_ind] in warning_list:
                sql = ''
                print(index, sample_ind)
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
            if len(df) > 0:
                valid_parse += 1
                sql_pair = {}
                sql_pair['id'] = index
                sql_pair['sample_ind'] = sample_ind
                sql_pair['sql'] = sql
                sql_pair['table'] = df
                sql_executable_count.append(sql_pair)
    
    sql_exec_df_output = merge_clean_and_format_df_dict(sql_exec_df)
    
    # Aggregate processed tables - Cell 29
    processed_table = {}
    processed_table['Base'] = wikitq_df_processed
    processed_table['Select_Row'] = filtered_tables_row
    processed_table['Select_Column'] = filtered_tables
    processed_table['RAG_20_5'] = RAG_20_5
    processed_table['Execute_SQL'] = sql_exec_df_output
    processed_table['Execute_SQL_count'] = sql_executable_count
    
    # Save processed tables - Cell 30
    np.save(f'{args.tmp_save_path}/processed_table.npy', processed_table)
    print(f"Saved processed tables to {args.tmp_save_path}/processed_table.npy")
    
    timeline['Step 9 - SQL Parsing & Execution'] = time.perf_counter() - _t9
    print(f"  [Timing] Step 9 - SQL Parsing & Execution: {timeline['Step 9 - SQL Parsing & Execution']:.2f}s")
    
    # ========================================================================
    # Step 10: Check Model Iteration - Cells 33, 35, 36
    # ========================================================================
    print("\n[Step 10] Running Check Model...")
    _t10 = time.perf_counter()
    
    # Initialize Check Model Data Sequence - Cell 33
    Check_Model_Data_Sequence = {}
    for key in ranked_result.keys():
        start_sequence = ranked_result[key]
        Check_Model_Data_Sequence[key] = {}
        Check_Model_Data_Sequence[key]['id'] = key
        Check_Model_Data_Sequence[key]['Sequence'] = start_sequence
        if start_sequence == ['Base'] or start_sequence == ['Execute_SQL']:
            Check_Model_Data_Sequence[key]['Terminated'] = True
            Check_Model_Data_Sequence[key]['Check_Status'] = False
            Check_Model_Data_Sequence[key]['Check_Score'] = 0.0
        else:
            Check_Model_Data_Sequence[key]['Terminated'] = False
            Check_Model_Data_Sequence[key]['Check_Status'] = False
            Check_Model_Data_Sequence[key]['Check_Score'] = 0.0
    
    for key in Check_Model_Data_Sequence.keys():
        data_entry = Prepare_Data_for_Operator_Sequence(
            key, Check_Model_Data_Sequence[key]['Sequence'],
            dataset, processed_table
        )
        Check_Model_Data_Sequence[key]['data_entry'] = data_entry
    
    # Load reranker model - Cell 35
    print("  Loading reranker model...")
    # 禁用多进程避免死锁，使用单 GPU 推理
    reranker_model = FlagReranker(
        args.check_model_path, 
        use_fp16=True,
        devices=[0]  # 只使用单个 GPU，避免多进程死锁
    )
    
    # Iterative check (3 loops) - Cell 36
    print("  Running iterative check (3 rounds)...")
    check_tau = args.check_tau
    for loop in range(3):
        print(f"    Round {loop + 1}/3...")
        updated_data = batch_rerank_scores(
            reranker_model, Check_Model_Data_Sequence, batch_size=16
        )
        Check_Model_Data_Sequence = updated_data
        
        # Check Terminal Status
        for key in Check_Model_Data_Sequence.keys():
            if Check_Model_Data_Sequence[key]['Terminated'] == True:
                continue
            else:
                if Check_Model_Data_Sequence[key]['Check_Status'] == True:
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
                        if terminated_flag == False:
                            data_entry = Prepare_Data_for_Operator_Sequence(
                                key, ROLLBACK_seq, dataset, processed_table
                            )
                            Check_Model_Data_Sequence[key]['data_entry'] = data_entry
                else:
                    raise ValueError("Check Status should be True when Terminated is False after reranking.")
    
    # Save Check Model Data Sequence - Cell 41
    np.save(
        f'{args.tmp_save_path}/Check_Model_Data_Sequence.npy',
        Check_Model_Data_Sequence
    )
    
    timeline['Step 10 - Check Model Iteration'] = time.perf_counter() - _t10
    print(f"  [Timing] Step 10 - Check Model Iteration: {timeline['Step 10 - Check Model Iteration']:.2f}s")
    # Explicitly release reranker and free VRAM
    try:
        import gc
        import torch
        del reranker_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        print("❗️❗️Error releasing reranker model and freeing VRAM❗️❗️")
        pass
    
    # ========================================================================
    # Step 11: Add Missing Execute_SQL - Cells 39, 40
    # ========================================================================
    print("\n[Step 11] Adding Missing Execute_SQL...")
    _t11 = time.perf_counter()
    
    # Cell 39
    SQL_list_final = []
    for index in range(len(dataset)):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        if sequence == [] or sequence.__contains__('Execute_SQL'):
            SQL_list_final.append(index)
    
    add_sql_list = list(
        set(SQL_list_final) - set(LLM_query_list['Execute_SQL']['index'])
    )
    
    print(f"  Adding {len(add_sql_list)} missing SQL queries...")
    add_sql_query_list = []
    for index in add_sql_list:
        prompt = build_wikitq_prompt_from_df(
            dataset, wikitq_df_processed[index], index,
            template_path='../prompts/sql_reason_wtq.txt',
            processed=True
        )
        add_sql_query_list.extend([prompt])
    
    # add_sql_response_list = response_vllm(
    #     llm, tokenizer, add_sql_query_list,
    #     sample_num=args.sql_sample_num,
    #     temperature=args.temperature,
    #     top_p=args.top_p
    # )
    add_sql_response_list, metrics_rows["Add_SQL"], summaries["Add_SQL"] = infer_prompts(
        add_sql_query_list,
        sample_num=args.sql_sample_num,
        temperature=args.temperature,
        top_p=args.top_p,
        llm_path=args.llm_path,
        concurrency=args.llm_concurrency
    )
    print(f"  Add SQL generation infer summary: {summaries['Add_SQL']}")

    # Parse and execute additional SQL - Cell 40
    sample_num = 3
    sql_exec_df = {}
    valid_parse = 0
    warning_list = []
    sql_executable_count = []
    
    for i in range(len(add_sql_list)):
        index = add_sql_list[i]
        sql_exec_df[index] = []
        for sample_ind in range(sample_num):
            original_text = add_sql_response_list[i][sample_ind]
            sql = fix_sql_query(
                response_text=original_text,
                table_df=wikitq_df_processed[index],
                table_title=table_titles[index]
            )
            if [index, sample_ind] in warning_list:
                sql = ''
                print(index, sample_ind)
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
            if len(df) > 0:
                valid_parse += 1
                sql_pair = {}
                sql_pair['id'] = index
                sql_pair['sample_ind'] = sample_ind
                sql_pair['sql'] = sql
                sql_pair['table'] = df
                sql_executable_count.append(sql_pair)
    
    sql_exec_df_output_new = merge_clean_and_format_df_dict(sql_exec_df)
    for index in sql_exec_df_output_new.keys():
        sql_exec_df_output[index] = sql_exec_df_output_new[index]
    processed_table['Execute_SQL'] = sql_exec_df_output
    
    # Save processed tables - Cell 42
    np.save(f'{args.tmp_save_path}/processed_table.npy', processed_table)
    
    timeline['Step 11 - Add Missing SQL'] = time.perf_counter() - _t11
    print(f"  [Timing] Step 11 - Add Missing SQL: {timeline['Step 11 - Add Missing SQL']:.2f}s")
    
    # ========================================================================
    # Step 12: Generate Final QA Prompts - Cell 45
    # ========================================================================
    print("\n[Step 12] Generating Final QA Prompts...")
    _t12 = time.perf_counter()
    
    prompt_list = []
    for index in range(len(dataset)):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        prompt = build_wikitq_prompt_from_df(
            dataset,
            Check_Model_Data_Sequence[index]['data_entry']['table'],
            index,
            template_path='../prompts/text_reason_wtq.txt',
            processed=True
        )
        if sequence == [] or sequence.__contains__('Execute_SQL'):
            evidence = table_to_str_sql(processed_table['Execute_SQL'][index])
            prompt = prompt + evidence
        prompt_list.append(prompt)
    
    timeline['Step 12 - Generate Final QA Prompts'] = time.perf_counter() - _t12
    print(f"  [Timing] Step 12 - Generate Final QA Prompts: {timeline['Step 12 - Generate Final QA Prompts']:.2f}s")
    
    # ========================================================================
    # Step 13: Execute Final QA and Evaluate - Cell 46
    # ========================================================================
    print("\n[Step 13] Executing Final QA...")
    _t13 = time.perf_counter()
    
    # qa_final = response_vllm(
    #     llm, tokenizer, prompt_list,
    #     sample_num=1,
    #     temperature=0,
    #     top_p=1
    # )
    qa_final, metrics_rows["Final_QA"], summaries["Final_QA"] = infer_prompts(
        prompt_list,
        sample_num=1,
        temperature=0,
        top_p=1,
        llm_path=args.llm_path,
        concurrency=max(4, min(8, args.llm_concurrency))
    )
    print(f"  Final QA generation infer summary: {summaries['Final_QA']}")

    ## shutdown
    mon.stop()
    per_avg, overall = mon.avg_util()

    # Create result dataframe
    wikitq_df = pd.DataFrame(dataset)
    wikitq_df['instruction'] = prompt_list
    wikitq_df['predict'] = qa_final
    wikitq_df['predict'] = [str(s) for s in qa_final]
    
    timeline['Step 13 - Final QA Generation'] = time.perf_counter() - _t13
    print(f"  [Timing] Step 13 - Final QA Generation: {timeline['Step 13 - Final QA Generation']:.2f}s")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    ## step 12: evaluate
    acc_all, error_index_all, format_error_index_all = evaluate_predictions(args.dataset_name, wikitq_df, dataset)
    print(f"Accuracy: {acc_all:.2f}%")

    total_time = time.perf_counter() - overall_start
    
    # Save results
    wikitq_df.to_csv(f'{args.tmp_save_path}/final_results.csv', index=False)
    print(f"Saved results to {args.tmp_save_path}/final_results.csv")
    
    # Save evaluation results
    with open(f'{args.tmp_save_path}/evaluation_results.json', 'w') as f:
        json.dump({
            'accuracy': acc_all,
            'error_indices': error_index_all,
            'format_error_indices': format_error_index_all,
            'total_samples': len(dataset)
        }, f, indent=2)
    
    # ========================================================================
    # Final Summary
    # ========================================================================

    # write metrics and summary into json files
    metrics_rows_f = f"{args.tmp_save_path}/metrics_n{args.first_n}_p{args.llm_concurrency}.json"
    summaries_f = f"{args.tmp_save_path}/summaries_n{args.first_n}_p{args.llm_concurrency}.json"
    with open(metrics_rows_f, 'w') as f:
        json.dump(metrics_rows, f, indent=2)
    with open(summaries_f, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Metrics and summaries saved to {metrics_rows_f} and {summaries_f}")
    total_latency = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for key, value in summaries.items():
        total_latency += value['batch_dur']
        total_prompt_tokens += value['total_prompt_tokens']
        total_completion_tokens += value['total_completion_tokens']
    print(f"Total latency: {total_latency:.2f} seconds")
    print(f"Total prompt tokens: {total_prompt_tokens}. Average prompt tokens: {total_prompt_tokens / len(dataset)}")
    print(f"Total completion tokens: {total_completion_tokens}. Average completion tokens: {total_completion_tokens / len(dataset)}")


    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Accuracy: {acc_all:.2f}%")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nTiming Breakdown:")
    print(f"{'-'*80}")
    for step_name, duration in timeline.items():
        percentage = (duration / total_time) * 100
        print(f"  {step_name:<45} {duration:>8.2f}s ({percentage:>5.1f}%)")
    print(f"{'-'*80}")
    print(f"Results saved to: {args.tmp_save_path}")
    print(f"{'='*80}\n")
    
    # Save timeline to file
    timeline_summary = {
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'accuracy': acc_all,
        'steps': timeline
    }

    print(f"Per-GPU avg util: {per_avg}")
    print(f"Overall avg util: {overall}")

    with open(f'{args.tmp_save_path}/timing_summary_n{args.first_n}_p{args.llm_concurrency}.json', 'w') as f:
        json.dump(timeline_summary, f, indent=2)
    print(f"Timing summary saved to {args.tmp_save_path}/timing_summary.json")


if __name__ == "__main__":
    main()
