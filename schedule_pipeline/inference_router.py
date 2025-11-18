import json
import pandas as pd
import pickle
import torch
from FlagEmbedding import FlagModel
import argparse
import os
from tqdm import tqdm
import time

def dataframe_to_llm_string(df: pd.DataFrame) -> str:
    """
    将 pandas DataFrame 转换为 LLM 更易读的 markdown 格式字符串。
    """
    header = "col : " + " | ".join(map(str, df.columns))
    rows = []
    for index, row in df.iterrows():
        row_values = [str(x) for x in row.values]
        row_str = "row " + str(index) + " : " + " | ".join(row_values)
        rows.append(row_str)
    return header + "\n" + "\n".join(rows)

def batch_inference(input_path, output_path, model_path):
    """
    使用微调后的模型进行批量推理，并将结果存入新的 pickle 文件。

    Args:
        input_path (str): 输入的 pickle 文件路径（不含标签）。
        output_path (str): 输出的 pickle 文件路径（包含推理结果）。
        model_path (str): 微调后模型的路径。
    """
    # 定义 instruction, 确保与训练时完全一致
    instruction = "What kind of operation should I take to better filter relevant tables to complete the QA? "

    # 加载模型
    try:
        print(f"正在从 {model_path} 加载模型...")
        model = FlagModel(model_path, 
                          query_instruction_for_retrieval=instruction, 
                          use_fp16=True)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 定义所有可能的标签
    ALL_LABELS = [
        'Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5', 'RAG_10_3'
    ]
    
    # 检测可用的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 预先编码所有标签
    print("正在编码所有候选标签...")
    label_embeddings_np = model.encode(ALL_LABELS)
    label_embeddings = torch.from_numpy(label_embeddings_np)
    print("标签编码完成。")

    # 加载推理数据
    try:
        with open(input_path, 'rb') as f:
            inference_data = pickle.load(f)
            if isinstance(inference_data, dict):
                inference_data = list(inference_data.values())
    except FileNotFoundError:
        print(f"错误: 输入文件未找到于 {input_path}")
        return
    except Exception as e:
        print(f"读取推理数据失败: {e}")
        return
        
    print(f"共找到 {len(inference_data)} 条数据进行推理。")
    
    # 准备所有查询
    queries_to_encode = []
    for item in inference_data:
        original_query = item['query']
        title = item['title']
        table_df = item['table']
        table_str = dataframe_to_llm_string(table_df)
        base_query_format = "Query: {} [SEP] Table Title: {} [SEP] Table: {}"
        model_query = base_query_format.format(original_query, title, table_str)
        queries_to_encode.append(model_query)

    # 批量编码所有查询
    print("正在批量编码所有查询...")
    query_embeddings_np = model.encode(queries_to_encode, batch_size=4, max_length=2048)
    query_embeddings = torch.from_numpy(query_embeddings_np)
    print("查询编码完成。")
    
    # 计算相似度
    print("正在计算相似度...")
    similarities_matrix = query_embeddings @ label_embeddings.T
    
    # 将结果整合
    results_data = []
    for i, item in enumerate(tqdm(inference_data, desc="整合结果")):
        similarities = similarities_matrix[i].cpu().tolist()
        label_scores = {label: score for label, score in zip(ALL_LABELS, similarities)}
        
        new_item = item.copy()
        new_item['result'] = label_scores
        results_data.append(new_item)
    
    print("相似度计算完成。")

    # 保存结果
    print(f"正在保存结果至: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'wb') as f_out:
        pickle.dump(results_data, f_out)
        
    print(f"推理完成，结果已成功保存。")

def main():
    """主函数，用于解析命令行参数并启动批量推理。"""
    parser = argparse.ArgumentParser(description="使用微调后的模型对表格数据进行批量分类推理。")
    parser.add_argument('--input_path', type=str, required=True, help="输入的 pickle 文件路径。")
    parser.add_argument('--model_path', type=str, required=True, help="微调后模型的路径。")
    parser.add_argument('--output_path', type=str, required=True, help="保存推理结果的 pickle 文件路径。")
    
    args = parser.parse_args()
    start_time = time.time()
    print("--- 启动批量推理脚本 ---")
    print(f"输入文件: {args.input_path}")
    print(f"模型路径: {args.model_path}")
    print(f"输出文件: {args.output_path}")
    
    batch_inference(args.input_path, args.output_path, args.model_path)
    end_time = time.time()
    print("--- 脚本执行完毕 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
if __name__ == '__main__':
    main()