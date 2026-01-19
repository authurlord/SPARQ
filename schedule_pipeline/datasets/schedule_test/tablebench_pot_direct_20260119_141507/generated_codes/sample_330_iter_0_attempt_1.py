import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 2003 and Work is "Road to Perdition" and Result is "Won"
award = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]['Award'].values[0]
print(f"Final Answer: {award}")