import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float by removing '%' and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the row with the highest percentage of national votes
max_vote_row = df.loc[df['% of national vote'].idxmax()]
year_max = max_vote_row['election']
max_vote = max_vote_row['% of national vote']

# Find the previous year's vote percentage
previous_year_idx = df.index[df['election'] == year_max][0] - 1
if previous_year_idx >= 0:
    prev_vote = df.iloc[previous_year_idx]['% of national vote']
else:
    prev_vote = 0  # If no previous year, assume 0

# Calculate the difference
difference = max_vote - prev_vote

print(f"Final Answer: {year_max}, {difference:.2f}")