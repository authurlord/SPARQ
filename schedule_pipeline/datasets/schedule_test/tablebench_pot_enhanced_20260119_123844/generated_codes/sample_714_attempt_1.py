import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'share of votes' to numeric by removing the '%' sign
df['share of votes'] = df['share of votes'].str.replace('%', '').astype(float)
# Find the election with the highest share of votes
max_vote_election = df.loc[df['share of votes'].idxmax(), 'election']
# Find the lowest share of votes
min_vote_share = df['share of votes'].min()
# Find the highest share of votes
max_vote_share = df['share of votes'].max()
# Calculate the difference
difference = max_vote_share - min_vote_share
print(f"Final Answer: {max_vote_election}, {difference:.1f}")