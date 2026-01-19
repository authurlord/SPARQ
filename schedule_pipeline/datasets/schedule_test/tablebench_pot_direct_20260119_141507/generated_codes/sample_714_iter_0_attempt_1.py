import pandas as pd

df = pd.read_csv('table.csv')
# Find the election with the highest and lowest share of votes
max_votes = df.loc[df['share of votes'].idxmax(), 'election']
min_votes = df.loc[df['share of votes'].idxmin(), 'election']
difference = df['share of votes'].max() - df['share of votes'].min()

print(f"Final Answer: {max_votes}, {difference:.1f}%")