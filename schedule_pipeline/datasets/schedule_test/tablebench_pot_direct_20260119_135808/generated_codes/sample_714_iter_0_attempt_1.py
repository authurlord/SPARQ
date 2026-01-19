import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'share of votes' to numeric by removing '%' and converting to float
df['share of votes'] = df['share of votes'].str.replace('%', '').astype(float)
# Find the election with the highest share of votes
max_vote_election = df.loc[df['share of votes'].idxmax(), 'election']
# Find the lowest share of votes
min_vote = df['share of votes'].min()
max_vote = df['share of votes'].max()
# Calculate the difference
difference = max_vote - min_vote
print(f"Final Answer: {max_vote_election}, {difference:.1f}")