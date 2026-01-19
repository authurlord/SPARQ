import pandas as pd

df = pd.read_csv('table.csv')
# Remove '%' and convert to float for comparison
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)
# Find the index of the maximum value
max_index = df['% of popular vote'].idxmax()
# Get the corresponding election year
highest_vote_year = df.loc[max_index, 'election']
print(f"Final Answer: {highest_vote_year}")