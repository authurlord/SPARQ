import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 2003, Work = "Road to Perdition", and Result = "Won"
winning_award = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]['Award'].values[0]
print(f"Final Answer: {winning_award}")