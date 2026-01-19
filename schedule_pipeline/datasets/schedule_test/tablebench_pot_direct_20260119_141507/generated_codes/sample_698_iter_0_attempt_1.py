import pandas as pd

df = pd.read_csv('table.csv')
# Find the election year with the highest percentage of popular vote
max_pop_vote_row = df.loc[df['% of popular vote'].idxmax()]
election_year = max_pop_vote_row['election']
print(f"Final Answer: {election_year}")