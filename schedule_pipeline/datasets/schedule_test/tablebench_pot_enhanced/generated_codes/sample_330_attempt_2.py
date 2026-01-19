import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for 2003 and work "Road to Perdition" with a win
winning_award = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]['Award'].iloc[0]
print(f"Final Answer: {winning_award}")