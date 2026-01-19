import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where rank is 4
rank_4_nation = df[df['rank'] == '4']['nation'].values[0]
print(f"Final Answer: {rank_4_nation}")