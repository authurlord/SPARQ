import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Transparency International
transparency_df = df[df['organization'] == 'transparency international']
# Find the row with the lowest rank (highest rank number means lower position)
highest_ranked_year = transparency_df.loc[transparency_df['rank'].astype(int).idxmin(), 'year']
print(f"Final Answer: {highest_ranked_year}")