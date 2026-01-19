import pandas as pd

df = pd.read_csv('table.csv')
# Find the jurisdiction with the highest 'percent for'
max_support_jurisdiction = df.loc[df['percent for'].idxmax(), 'jurisdiction']
print(f"Final Answer: {max_support_jurisdiction}")