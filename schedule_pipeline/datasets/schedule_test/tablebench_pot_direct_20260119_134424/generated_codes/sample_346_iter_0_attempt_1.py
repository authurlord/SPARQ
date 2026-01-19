import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'spanish' column to integer for comparison
df['spanish'] = pd.to_numeric(df['spanish'])
# Count municipalities with Spanish speakers >= 40,000
count = (df['spanish'] >= 40000).sum()
print(f"Final Answer: {count}")