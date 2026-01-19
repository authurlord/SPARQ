import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with rank 4
nation_rank_4 = df[df['rank'] == '4']['nation'].values[0]
print(f"Final Answer: {nation_rank_4}")