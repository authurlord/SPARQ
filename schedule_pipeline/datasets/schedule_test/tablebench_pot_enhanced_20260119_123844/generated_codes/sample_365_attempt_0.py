import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to numeric for comparison
df['percent for'] = pd.to_numeric(df['percent for'])
# Count jurisdictions with more than 70% in favor
count = (df['percent for'] > 70).sum()
print(f"Final Answer: {count}")