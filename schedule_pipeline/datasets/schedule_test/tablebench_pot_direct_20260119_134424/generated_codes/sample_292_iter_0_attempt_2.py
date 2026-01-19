import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Sunny Morning won in 2017
filtered_row = df[(df['Year'] == '2017') & (df['Nominated Work'] == 'Sunny Morning') & (df['Result'] == 'Won')]
award = filtered_row['Award'].iloc[0]
edition = filtered_row['Notes'].iloc[0].split('edition #')[1].split(' ')[0] if 'edition #' in filtered_row['Notes'].iloc[0] else None
print(f"Final Answer: {award}, {edition}")