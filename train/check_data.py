import pickle
from collections import Counter

data = pickle.load(open('inference_results.pkl', 'rb'))
valid = [d for d in data if d.get('label')]

print(f'总数据: {len(data)}')
print(f'有标签的数据: {len(valid)}')
print(f'\n前5个有标签的样本:')

for i, d in enumerate(valid[:5]):
    print(f'  样本{i}: label={d["label"]}, label_count={len(d["label"])}')

print(f'\n所有有标签数据的标签分布:')
all_labels = []
label_counts = []
for d in valid:
    all_labels.extend(d['label'])
    label_counts.append(len(d['label']))

print(f'标签分布: {Counter(all_labels)}')
print(f'标签数量分布: {Counter(label_counts)}')

# 统计标签数量<5的数据
small_label_data = [d for d in valid if len(d['label']) < 5]
print(f'\n标签数量<5的数据: {len(small_label_data)}')

