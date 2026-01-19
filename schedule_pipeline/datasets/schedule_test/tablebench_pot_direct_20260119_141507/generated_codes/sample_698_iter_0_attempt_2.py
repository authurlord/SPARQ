import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of popular vote' to numeric, removing the '%' sign
df['% of popular vote'] = df['% of popular vote'].str.rstrip('%').astype(float)
# Find the election year with the highest percentage of popular vote
max_vote_row = df.loc[df['% of popular vote'].idxmax()]
print(f"Final Answer: {max_vote_row['election']}")