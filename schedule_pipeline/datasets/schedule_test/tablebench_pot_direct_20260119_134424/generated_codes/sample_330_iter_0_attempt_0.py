import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 2003, Work = "Road to Perdition", and Result = "Won"
winner_award = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]['Award'].iloc[0]
print(f"Final Answer: {winner_award}")