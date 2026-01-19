import pandas as pd

df = pd.read_csv('table.csv')
# Find the election year with the highest percentage of party list votes
max_party_list_votes = df['% of party list votes'].idxmax()
election_year = df.loc[max_party_list_votes, 'Election year']
print(f"Final Answer: {election_year}")