import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with rank 3
nation_rank_3 = df[df['rank'] == '3']['nation'].iloc[0]
print(f"Final Answer: {nation_rank_3}")