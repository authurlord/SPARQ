import pandas as pd

df = pd.read_csv('table.csv')

# Convert '% of national vote' to float by removing '%' and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the year with the highest % of national vote
max_vote_row = df.loc[df['% of national vote'].idxmax()]
max_year = max_vote_row['election']
max_vote = max_vote_row['% of national vote']

# Find the previous year's vote percentage (the year before max_year)
prev_year = str(int(max_year) - 1)
prev_vote_row = df[df['election'] == prev_year]
prev_vote = prev_vote_row['% of national vote'].values[0] if not prev_vote_row.empty else 0

# Calculate the difference
difference = max_vote - prev_vote

print(f"Final Answer: {max_year}, {difference:.2f}")