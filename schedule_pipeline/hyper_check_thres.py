from utils.schedule_utils import load_data_split,table_to_str,table_to_str_sql,find_intersection_and_add_row_id,Prepare_Data_for_Operator_Sequence,format_document,batch_rerank_scores,ROLLBACK,merge_clean_and_format_df_dict,retrieve_rows_by_subtables,process_error_analysis_list
import pickle
import numpy as np
import pandas as pd
import os
import json
from FlagEmbedding import FlagReranker
from tqdm import tqdm
from typing import Dict, Any
from utils.evaluator import Evaluator
from utils.prompt_generate import build_wikitq_prompt_from_df,evaluate_predictions,filter_dataframe_from_responses,fix_sql_query,match_subtables,retrieve_rows_by_subtables,build_tab_fact_prompt_from_df
from utils.async_llm import infer_prompts

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from time import sleep
import argparse
from vllm.lora.request import LoRARequest
import json
from tqdm import tqdm
import os 

ALL_LABELS = ['Base', 'Execute_SQL', 'RAG', 'Select_Column', 'Select_Row']
def parse_valid_route(input):
    parse = eval(input)[0]
    eval_parse = eval(parse)
    ## 如果有非法字符
    return [e for e in eval_parse if e in ALL_LABELS]

## 参数记录 for wikitq ablation training-free router
N_PARALLEL = 32 ## only for convert_df_type_parallel.py of data preprocess，一次性离线的
embedding_base_model_path = '/data/workspace/yanmy/models/bge-m3'
llm_path = '/data/workspace/yanmy/models/Qwen3-4B-Instruct-2507'
router_model_path = '/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/' ## fine-tuned router model path
llm_path = ''
check_moedl_path = '/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/' ## check model path
tmp_save_path = 'datasets/schedule_test/wikitq_4B_hyper' ## 临时存储路径
dataset_name = 'wikitq'
split = 'test'
tau = 0.82
check_tau = 0.8
ALL_LABELS = [
    'Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5', 
]

RAG_20_5 = np.load('datasets/schedule_test/wikitq/Hybrid_Retrieve_output_update.npy',allow_pickle=True).item()
wikitq_df_processed = np.load('datasets/schedule_test/wikitq/wikitq_df_processed.npy',allow_pickle=True).item()
rewrite_query_list = np.load('datasets/schedule_test/wikitq/rewrite_query.npy',allow_pickle=True)
training_free_router = pd.read_csv('../datasets/ablation/wikitq_test_training_free_router_output_4B.csv',index_col=0)
training_free_router = [parse_valid_route(x) for x in training_free_router['predict'].tolist()]
filtered_table_column = np.load('../datasets/pipeline/filtered_tables_col_test.npy',allow_pickle=True).item()
filtered_table_row = np.load('../datasets/pipeline/filtered_tables_row_test.npy',allow_pickle=True).item()
sql_exec_df_output = np.load('../datasets/pipeline/sql_exec_df_output_test.npy',allow_pickle=True).item()
sql_executable_count = np.load('../datasets/pipeline/wikitq_test_sql_executable_count_30B.npy',allow_pickle=True)
processed_table = {} ## processed_table包括了inference LLM后所有结果的聚合,key 与 method_key 保持一致
processed_table['Base'] = wikitq_df_processed
processed_table['Select_Row'] = filtered_table_row
processed_table['Select_Column'] = filtered_table_column
processed_table['RAG'] = RAG_20_5
processed_table['RAG_20_5'] = RAG_20_5
processed_table['Execute_SQL'] = sql_exec_df_output ## SQL聚合结果！
processed_table['Execute_SQL_count'] = sql_executable_count ## SQL单个结

dataset = load_data_split(dataset_name,split)

Check_Model_Data = np.load('datasets/schedule_test/wikitq/Check_Model_Data_Ablation.npy',allow_pickle=True).item()

metrics = {}

for key in Check_Model_Data.keys():
    Check_Model_Data_Sequence = Check_Model_Data[key]
    prompt_list = []
    for index in range(len(dataset)):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        prompt = build_wikitq_prompt_from_df(dataset,Check_Model_Data_Sequence[index]['data_entry']['table'],index,template_path='../prompts/text_reason_wtq.txt',processed=True)
        if sequence==[] or sequence.__contains__('Execute_SQL'):
            # try:
            evidence = table_to_str_sql(processed_table['Execute_SQL'][index])
            prompt = prompt + evidence
            # print(f'SQL at index {index}')
        prompt_list.append(prompt)
    print(f'Check at tau {key}')
    qa_final = infer_prompts(prompt_list,temperature=0, top_p=1, sample_num=1,concurrency=256)
    wikitq_df = pd.DataFrame(dataset)
    wikitq_df['instruction'] = prompt_list
    wikitq_df['predict'] = qa_final[0]
    ## step 12: evaluate
    wikitq_df['predict'] = [str(s) for s in qa_final[0]]
    acc_all, error_index_all, format_error_index_all = evaluate_predictions(dataset_name, wikitq_df, dataset)

    metrics[key] = {
        'accuracy': acc_all,
        'error_index': error_index_all,
        'format_error_index': format_error_index_all,
        'predict' : qa_final
    }

    print(f'Check Model at tau {key} Accuracy: {acc_all}')
np.save('metrics_hyper_tau.npy',metrics)

