import pandas as pd

df = pd.read_csv('table.csv')
# Find the maximum and minimum share of votes and their corresponding elections
max_votes = df.loc[df['share of votes'].idxmax(), 'election']
min_votes = df.loc[df['share of votes'].idxmin(), 'election']
max_share = df['share of votes'].max()
min_share = df['share of votes'].min()
difference = max_share - min_share

print(f"Final Answer: {max_votes}, {min_votes}, {difference:.1f}")