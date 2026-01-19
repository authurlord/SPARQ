import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific work, year, and result
filtered_df = df[(df['Year'] == '2017') & (df['Nominated Work'] == 'Sunny Morning') & (df['Result'] == 'Won')]
# Extract the award and edition from the notes
award = filtered_df['Award'].iloc[0]
edition = filtered_df['Notes'].iloc[0].split('edition #')[1].split(' (2017)')[0] if 'edition #' in filtered_df['Notes'].iloc[0] else ''
print(f"Final Answer: {award}, {edition}")