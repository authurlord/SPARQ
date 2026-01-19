import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with rank 4
fourth_ranked_nation = df[df['rank'] == '4']['nation'].values[0]
print(f"Final Answer: {fourth_ranked_nation}")