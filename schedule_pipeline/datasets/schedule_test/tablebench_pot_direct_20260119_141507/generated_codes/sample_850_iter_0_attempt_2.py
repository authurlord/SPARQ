import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float and extract relevant columns
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the year with the highest % of national vote
max_vote_row = df.loc[df['% of national vote'].idxmax()]
max_year = max_vote_row['election']
max_percentage = max_vote_row['% of national vote']

# Get the previous year's percentage
previous_row = df[df['election'] < max_year].iloc[-1]
previous_percentage = previous_row['% of national vote']

# Calculate the difference
difference = max_percentage - previous_percentage

print(f"Final Answer: {max_year}, {difference:.2f}")