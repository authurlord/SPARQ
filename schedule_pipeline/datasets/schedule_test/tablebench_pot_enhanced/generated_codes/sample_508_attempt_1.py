import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to numeric for comparison
df['percent for'] = pd.to_numeric(df['percent for'])
# Find the jurisdiction with the highest Prohibition Support Rate
max_support_jurisdiction = df.loc[df['percent for'].idxmax(), 'jurisdiction']
print(f"Final Answer: {max_support_jurisdiction}")