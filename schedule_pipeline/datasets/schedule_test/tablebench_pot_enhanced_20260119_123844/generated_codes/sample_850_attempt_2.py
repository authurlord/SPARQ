import pandas as pd

df = pd.read_csv('table.csv')

# Convert '% of national vote' to float for comparison
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the row with the highest % of national vote
max_vote_row = df.loc[df['% of national vote'].idxmax()]

# Get the year and the percentage
max_year = max_vote_row['election']
max_percentage = max_vote_row['% of national vote']

# Find the previous year's percentage
prev_year_idx = df[df['election'] == max_year].index[0] - 1
if prev_year_idx >= 0:
    prev_percentage = df.iloc[prev_year_idx]['% of national vote']
else:
    prev_percentage = 0  # No previous year

# Calculate the difference
difference = max_percentage - prev_percentage

print(f"Final Answer: {max_year}, {difference:.2f}")