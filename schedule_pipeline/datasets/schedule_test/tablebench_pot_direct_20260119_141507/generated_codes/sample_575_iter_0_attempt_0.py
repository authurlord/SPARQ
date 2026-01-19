import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is 2005
df_2005 = df[df['year'] == '2005']
# Find the minimum rank (highest-rated indicator)
highest_ranked_rank = df_2005['rank'].min()
print(f"Final Answer: {highest_ranked_rank}")