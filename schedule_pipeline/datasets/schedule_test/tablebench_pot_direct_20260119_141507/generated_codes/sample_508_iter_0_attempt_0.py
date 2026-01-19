import pandas as pd

df = pd.read_csv('table.csv')
# Find the jurisdiction with the highest percentage in favor of prohibition
max_support_jurisdiction = df.loc[df['percent for'].idxmax(), 'jurisdiction']
print(f"Final Answer: {max_support_jurisdiction}")