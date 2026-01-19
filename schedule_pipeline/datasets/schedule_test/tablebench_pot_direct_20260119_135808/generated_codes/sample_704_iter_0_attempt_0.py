import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert '% of party list votes' to numeric
df['% of party list votes'] = pd.to_numeric(df['% of party list votes'], errors='coerce')

# Find the election year with the highest percentage of party list votes
max_vote_year = df.loc[df['% of party list votes'].idxmax(), 'Election year']

print(f"Final Answer: {max_vote_year}")