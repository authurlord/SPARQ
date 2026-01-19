import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is 2005
filtered_df = df[df['year'] == '2005']
# Find the minimum rank (highest-rated indicator)
highest_rated_rank = filtered_df['rank'].min()
print(f"Final Answer: {highest_rated_rank}")