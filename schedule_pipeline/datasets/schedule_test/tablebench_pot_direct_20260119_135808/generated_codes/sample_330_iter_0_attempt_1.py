import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the year 2003, work "Road to Perdition", and result "Won"
winning_award = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]['Award'].iloc[0]
print(f"Final Answer: {winning_award}")