import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'occurrence' to integer type
df['occurrence'] = df['occurrence'].astype(int)
# Count how many have occurrence greater than 1
count_occurrence_gt_1 = df[df['occurrence'] > 1].shape[0]
print(f"Final Answer: {count_occurrence_gt_1}")