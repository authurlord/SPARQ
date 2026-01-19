import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert '% of national vote' to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)
# Find the year with the highest percentage of national votes
max_vote_year = df.loc[df['% of national vote'].idxmax()]
# Find the previous year's percentage (by sorting and getting the prior row)
df_sorted = df.sort_values('election')
max_index = df_sorted.index.get_loc(max_vote_year.name)
previous_year_vote = df_sorted.iloc[max_index - 1]['% of national vote'] if max_index > 0 else 0
# Calculate the difference
difference = max_vote_year['% of national vote'] - previous_year_vote
print(f"Final Answer: {max_vote_year['election']}, {difference:.2f}")