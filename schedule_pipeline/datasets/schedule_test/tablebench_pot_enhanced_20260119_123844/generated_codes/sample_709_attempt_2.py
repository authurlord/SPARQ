import pandas as pd

df = pd.read_csv('table.csv')
# Find the party with the maximum total
max_party = df.loc[df['total'].idxmax(), 'party']
print(f"Final Answer: {max_party}")