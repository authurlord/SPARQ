import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Transparency International
transparency_df = df[df['organization'] == 'transparency international']
# Find the row with the lowest rank (highest rank number)
highest_ranked_row = transparency_df.loc[transparency_df['rank'].astype(int).idxmin()]
# Extract the year
year = highest_ranked_row['year']
print(f"Final Answer: {year}")