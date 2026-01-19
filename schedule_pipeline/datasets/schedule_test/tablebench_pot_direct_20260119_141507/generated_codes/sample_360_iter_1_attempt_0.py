import pandas as pd

df = pd.read_csv('table.csv')
# Convert Gold column to numeric to ensure proper comparison
df['Gold'] = pd.to_numeric(df['Gold'], errors='coerce')
# Count nations with at least one gold medal (Gold > 0)
gold_count = df[df['Gold'] > 0].shape[0]
print(f"Final Answer: {gold_count}")