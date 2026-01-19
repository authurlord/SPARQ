import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'spanish' column to integers
df['spanish'] = df['spanish'].astype(int)
# Count municipalities with Spanish population >= 40,000
count = (df['spanish'] >= 40000).sum()
print(f"Final Answer: {count}")