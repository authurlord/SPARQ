import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to float and find the maximum
max_support_rate = df['percent for'].max()
jurisdiction_with_max_support = df.loc[df['percent for'].idxmax(), 'jurisdiction']
print(f"Final Answer: {jurisdiction_with_max_support}")