import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with rank 3
third_rank_nation = df[df['rank'] == '3']['nation'].values[0]
print(f"Final Answer: {third_rank_nation}")