import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific work, year, and result
filtered_df = df[(df['Year'] == '2017') & (df['Nominated Work'] == 'Sunny Morning') & (df['Result'] == 'Won')]
# Extract the award and notes
award = filtered_df['Award'].iloc[0]
edition = filtered_df['Notes'].iloc[0]
print(f"Final Answer: {award}, {edition}")