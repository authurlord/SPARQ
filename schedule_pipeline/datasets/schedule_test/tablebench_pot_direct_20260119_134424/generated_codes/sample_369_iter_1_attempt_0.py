import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'semifinalists' column to integer
df['semifinalists'] = pd.to_numeric(df['semifinalists'], errors='coerce')
# Count countries with at least one semifinalist
count_with_semifinalists = (df['semifinalists'] >= 1).sum()
print(f"Final Answer: {count_with_semifinalists}")