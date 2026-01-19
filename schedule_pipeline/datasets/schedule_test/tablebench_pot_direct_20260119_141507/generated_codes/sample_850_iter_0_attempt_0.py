import pandas as pd

df = pd.read_csv('table.csv')
# Convert % of national vote to numeric (removing % sign)
df['% of national vote'] = df['% of national vote'].str.rstrip('%').astype(float)

# Find the row with the maximum % of national vote
max_vote_row = df.loc[df['% of national vote'].idxmax()]
max_year = max_vote_row['election']

# Get the previous year's value
prev_year_index = df[df['election'] < max_year].index[-1]
prev_year_vote = df.loc[prev_year_index, '% of national vote']

# Calculate the difference
difference = max_vote_row['% of national vote'] - prev_year_vote

print(f"Final Answer: {max_year}, {difference:.2f}")