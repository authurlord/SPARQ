import pandas as pd

df = pd.read_csv('table.csv')
# Convert population to integer
df['population'] = df['population'].astype(int)
# Count countries with population > 40 million
count = (df['population'] > 40000000).sum()
print(f"Final Answer: {count}")