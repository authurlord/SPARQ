import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Transparency International
transparency_df = df[df['organization'] == 'transparency international']
# Find the row with the lowest rank (highest rank number means worse performance)
highest_ranked_index = transparency_df.loc[transparency_df['rank'].astype(int).idxmin()]
year = highest_ranked_index['year']
print(f"Final Answer: {year}")