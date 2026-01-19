import pandas as pd

df = pd.read_csv('table.csv')
# Find the election year with the highest percentage of party list votes
max_party_list_votes = df['% of party list votes'].max()
year_with_max_votes = df[df['% of party list votes'] == max_party_list_votes]['Election year'].values[0]
print(f"Final Answer: {year_with_max_votes}")