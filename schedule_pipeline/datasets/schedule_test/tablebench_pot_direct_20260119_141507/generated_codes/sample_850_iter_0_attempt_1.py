import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of national vote' to float for numerical comparison
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)

# Find the row with the maximum % of national vote
max_vote_row = df.loc[df['% of national vote'].idxmax()]
max_year = max_vote_row['election']
max_percentage = max_vote_row['% of national vote']

# Get the previous year's percentage
prev_row = df[df['election'] < max_year].iloc[-1]
prev_percentage = prev_row['% of national vote']

# Calculate the difference
difference = max_percentage - prev_percentage

print(f"Final Answer: {max_year}, {difference:.2f}")