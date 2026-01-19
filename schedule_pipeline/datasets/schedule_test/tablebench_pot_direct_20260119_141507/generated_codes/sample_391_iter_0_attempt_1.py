import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of rows where 'occurrence' > 1
count_occurrence_gt_1 = df[df['occurrence'] > 1].shape[0]
print(f"Final Answer: {count_occurrence_gt_1}")