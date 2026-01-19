import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float by removing '%' and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the row with the highest percentage of national votes
max_vote_row = df.loc[df['% of national vote'].idxmax()]
year_max = max_vote_row['election']
max_vote = max_vote_row['% of national vote']

# Find the previous year's vote percentage (by sorting the election years)
df_sorted = df.sort_values('election').reset_index()
max_index = df_sorted[df_sorted['election'] == year_max].index[0]
prev_year_row = df_sorted.iloc[max_index - 1] if max_index > 0 else None

if prev_year_row is not None:
    prev_vote = prev_year_row['% of national vote']
    difference = max_vote - prev_vote
else:
    difference = 0  # If no previous year

print(f"Final Answer: {year_max}, {difference:.2f}")