import pandas as pd

df = pd.read_csv('table.csv')
# Remove the '%' and convert to float
df['share of votes'] = df['share of votes'].str.replace('%', '').astype(float)

# Find the maximum and minimum share of votes
max_votes = df['share of votes'].max()
min_votes = df['share of votes'].min()
max_election = df.loc[df['share of votes'] == max_votes, 'election'].values[0]
min_election = df.loc[df['share of votes'] == min_votes, 'election'].values[0]

difference = max_votes - min_votes

print(f"Final Answer: {max_election}, {difference:.1f}")