import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'semifinalists' column to integer
df['semifinalists'] = pd.to_numeric(df['semifinalists'], errors='coerce')
# Count countries with at least one semifinalist
count = (df['semifinalists'] > 0).sum()
print(f"Final Answer: {count}")