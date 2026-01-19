import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'share of votes' to float by removing '%' and converting to float
df['share of votes'] = df['share of votes'].str.replace('%', '').astype(float)

# Find the election with the highest and lowest share of votes
max_vote_election = df.loc[df['share of votes'].idxmax(), 'election']
min_vote_election = df.loc[df['share of votes'].idxmin(), 'election']
max_vote = df['share of votes'].max()
min_vote = df['share of votes'].min()
difference = max_vote - min_vote

print(f"Final Answer: {max_vote_election}, {min_vote_election}, {difference:.1f}")