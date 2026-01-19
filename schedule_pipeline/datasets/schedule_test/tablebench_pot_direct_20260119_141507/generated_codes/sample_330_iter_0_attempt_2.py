import pandas as pd

df = pd.read_csv('table.csv')
# Filter for year 2003 and work "Road to Perdition" with result "Won"
award = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]['Award'].values[0]
print(f"Final Answer: {award}")