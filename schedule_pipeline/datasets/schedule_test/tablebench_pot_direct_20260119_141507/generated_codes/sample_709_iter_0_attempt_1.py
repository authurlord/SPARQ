import pandas as pd

df = pd.read_csv('table.csv')
# Find the party with the highest total seats
max_total_party = df.loc[df['total'].idxmax(), 'party']
print(f"Final Answer: {max_total_party}")