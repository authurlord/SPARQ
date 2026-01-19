import pandas as pd

df = pd.read_csv('table.csv')
# Find the election with the highest and lowest share of votes
max_vote_row = df.loc[df['share of votes'].idxmax()]
min_vote_row = df.loc[df['share of votes'].idxmin()]

highest_election = max_vote_row['election']
lowest_election = min_vote_row['election']
difference = max_vote_row['share of votes'] - min_vote_row['share of votes']

print(f"Final Answer: {highest_election}, {lowest_election}, {difference:.1f}")