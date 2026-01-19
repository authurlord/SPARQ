import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'share of votes' to float by removing the '%' sign and converting to float
df['share of votes'] = df['share of votes'].str.replace('%', '').astype(float)

# Find the election with the highest share of votes
max_vote_election = df.loc[df['share of votes'].idxmax()]
# Find the election with the lowest share of votes
min_vote_election = df.loc[df['share of votes'].idxmin()]

# Calculate the difference
difference = max_vote_election['share of votes'] - min_vote_election['share of votes']

print(f"Final Answer: {max_vote_election['election']}, {difference:.1f}")