import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'occurrence' to numeric, coercing errors to NaN if any
df['occurrence'] = pd.to_numeric(df['occurrence'], errors='coerce')
# Count how many have occurrence > 1
count_occurrence_gt_1 = df[df['occurrence'] > 1].shape[0]
print(f"Final Answer: {count_occurrence_gt_1}")