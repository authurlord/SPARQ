import pandas as pd

df = pd.read_csv('table.csv')
# Find the season with the lowest (highest) rank value
min_rank_row = df.loc[df['rank'].idxmin()]
highest_rank_season = min_rank_row['season']
print(f"Final Answer: {highest_rank_season}")