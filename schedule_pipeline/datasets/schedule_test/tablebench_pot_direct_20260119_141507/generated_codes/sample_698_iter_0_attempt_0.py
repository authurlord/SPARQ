import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of popular vote' to float and find the maximum
max_vote_percentage = df['% of popular vote'].str.replace('%', '').astype(float)
max_index = max_vote_percentage.idxmax()
election_year = df.loc[max_index, 'election']
print(f"Final Answer: {election_year}")