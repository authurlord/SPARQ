import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of party list votes' to numeric, coercing errors to NaN
df['% of party list votes'] = pd.to_numeric(df['% of party list votes'], errors='coerce')
# Find the index of the maximum value in '% of party list votes'
max_index = df['% of party list votes'].idxmax()
# Get the corresponding election year
highest_election_year = df.loc[max_index, 'Election year']
print(f"Final Answer: {highest_election_year}")