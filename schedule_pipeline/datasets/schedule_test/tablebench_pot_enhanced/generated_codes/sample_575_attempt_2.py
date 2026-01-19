import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2005
df_2005 = df[df['year'] == '2005']
# Find the minimum rank (best rank) among indicators in 2005
highest_rank = df_2005['rank'].min()
print(f"Final Answer: {highest_rank}")