import pandas as pd

df = pd.read_csv('table.csv')
# Convert population to integer for comparison
df['population'] = pd.to_numeric(df['population'])
# Count countries with population > 40 million
count = (df['population'] > 40000000).sum()
print(f"Final Answer: {count}")