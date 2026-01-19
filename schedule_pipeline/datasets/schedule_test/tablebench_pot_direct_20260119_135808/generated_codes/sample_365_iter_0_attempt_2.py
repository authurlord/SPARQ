import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to float
df['percent for'] = pd.to_numeric(df['percent for'])
# Filter jurisdictions with more than 70% in favor
count = df[df['percent for'] > 70].shape[0]
print(f"Final Answer: {count}")