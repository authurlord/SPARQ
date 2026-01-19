import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' column to float for numerical comparison
df['percent for'] = df['percent for'].astype(float)
# Filter jurisdictions with more than 70% in favor
count = df[df['percent for'] > 70].shape[0]
print(f"Final Answer: {count}")