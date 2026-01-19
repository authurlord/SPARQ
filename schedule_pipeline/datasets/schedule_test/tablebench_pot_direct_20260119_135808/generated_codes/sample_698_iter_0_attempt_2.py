import pandas as pd

df = pd.read_csv('table.csv')
# Remove '%' and convert to float for comparison
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)
# Find the election year with the highest percentage
max_vote_year = df.loc[df['% of popular vote'].idxmax(), 'election']
print(f"Final Answer: {max_vote_year}")