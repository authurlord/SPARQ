import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where rank is 3
third_rank_nation = df[df['rank'] == '3']['nation'].iloc[0]
print(f"Final Answer: {third_rank_nation}")