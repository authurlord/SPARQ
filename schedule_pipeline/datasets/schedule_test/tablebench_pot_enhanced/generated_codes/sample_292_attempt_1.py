import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Sunny Morning, won in 2017
filtered_row = df[(df['Year'] == '2017') & (df['Nominated Work'] == 'Sunny Morning') & (df['Result'] == 'Won')]
award = filtered_row['Award'].values[0]
edition = filtered_row['Notes'].values[0]
print(f"Final Answer: {award}, {edition}")