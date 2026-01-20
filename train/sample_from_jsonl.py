import json
import random
from collections import Counter

def calculate_diversity_score(labels_list):
    """
    计算标签分布的多样性得分
    使用熵来衡量多样性
    """
    if not labels_list:
        return 0
    
    counter = Counter(labels_list)
    total = len(labels_list)
    entropy = 0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * (p ** 0.5)  # 使用修改的熵公式，更倾向于均匀分布
    return entropy

def sample_data_with_diversity(data_list, sample_size, prefer_small_labels=True):
    """
    采样数据，优先选择标签数量小于5且分布多样的数据
    
    Args:
        data_list: 数据列表
        sample_size: 采样数量
        prefer_small_labels: 是否优先选择标签数量较少的数据
    """
    # 为每个数据添加标签数量信息
    for item in data_list:
        item['label_count'] = len(item['pos'])
    
    if len(data_list) <= sample_size:
        return data_list
    
    # 如果优先选择标签数量小于5的数据
    if prefer_small_labels:
        # 分为两组：标签数量<5 和 >=5
        small_label_data = [item for item in data_list if item['label_count'] < 5]
        large_label_data = [item for item in data_list if item['label_count'] >= 5]
        
        print(f"标签数量<5的数据: {len(small_label_data)}, 标签数量>=5的数据: {len(large_label_data)}")
        
        # 优先从标签数量<5的数据中采样
        if len(small_label_data) >= sample_size:
            candidate_data = small_label_data
        else:
            # 如果标签数量<5的数据不够，补充一些标签数量>=5的数据
            candidate_data = small_label_data + large_label_data
    else:
        candidate_data = data_list
    
    # 使用贪心算法选择多样性最好的样本
    sampled_data = []
    remaining_data = candidate_data.copy()
    
    # 首先随机选择一个样本作为起点
    random.shuffle(remaining_data)
    sampled_data.append(remaining_data.pop(0))
    
    # 迭代选择剩余样本
    while len(sampled_data) < sample_size and remaining_data:
        # 计算当前已选样本的标签分布
        current_labels = []
        for item in sampled_data:
            current_labels.extend(item['pos'])
        
        # 为每个候选样本计算得分
        best_score = -float('inf')
        best_idx = 0
        
        for idx, item in enumerate(remaining_data):
            # 计算加入该样本后的多样性得分
            temp_labels = current_labels + item['pos']
            diversity_score = calculate_diversity_score(temp_labels)
            
            # 如果优先选择标签数量少的，给予额外加分
            label_count_bonus = 0
            if prefer_small_labels and item['label_count'] < 5:
                label_count_bonus = 0.5
            
            total_score = diversity_score + label_count_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
        
        sampled_data.append(remaining_data.pop(best_idx))
    
    # 打印采样后的标签分布统计
    all_labels = []
    label_counts = []
    for item in sampled_data:
        all_labels.extend(item['pos'])
        label_counts.append(item['label_count'])
    
    print(f"采样 {len(sampled_data)} 条数据")
    print(f"标签分布: {Counter(all_labels)}")
    print(f"标签数量分布: {Counter(label_counts)}")
    print(f"平均标签数量: {sum(label_counts) / len(label_counts):.2f}")
    
    return sampled_data

def sample_and_save(input_path, output_path, sample_size):
    """
    从输入的 jsonl 文件中采样并保存到输出文件
    """
    # 读取所有数据
    data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    print(f"总数据量: {len(data_list)}")
    
    # 采样
    sampled_data = sample_data_with_diversity(data_list, sample_size, prefer_small_labels=True)
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            # 移除临时添加的字段
            if 'label_count' in item:
                del item['label_count']
            f.write(json.dumps(item) + '\n')
    
    print(f"已保存到: {output_path}\n")

if __name__ == '__main__':
    # 设置随机种子以保证可重复性
    random.seed(42)
    
    # 输入文件路径
    input_file_path = '/data/workspace/yanmy/HybridRAG/H-STAR/router/finetune_data.jsonl'
    
    # 生成不同采样数量的训练数据
    sample_sizes = [20, 100, 400, 1000]
    
    for size in sample_sizes:
        output_file_path = f'train_{size}.jsonl'
        print(f"\n{'#'*80}")
        print(f"# 生成 {size} 条训练数据")
        print(f"{'#'*80}\n")
        sample_and_save(input_file_path, output_file_path, sample_size=size)

