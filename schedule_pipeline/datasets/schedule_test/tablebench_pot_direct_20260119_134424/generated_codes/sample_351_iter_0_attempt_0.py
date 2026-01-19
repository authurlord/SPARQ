import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float
df['%'] = df['%'].astype(float)
# Filter parties with more than 10% votes
high_vote_parties = df[df['%'] > 10]
# Count the number of such parties
count = len(high_vote_parties)
print(f"Final Answer: {count}")