import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2005
df_2005 = df[df['year'] == '2005']
# Find the minimum rank (highest-rated) among 2005 data
highest_rank_2005 = df_2005['rank'].min()
print(f"Final Answer: {highest_rank_2005}")