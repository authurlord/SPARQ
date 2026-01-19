import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'Transparency International'
transparency_df = df[df['organization'] == 'transparency international']
# Find the row with the lowest rank (highest position)
highest_ranked = transparency_df.loc[transparency_df['rank'].idxmin()]
# Extract the year
year = highest_ranked['year']
print(f"Final Answer: {year}")