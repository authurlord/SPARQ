import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific year, work, and win result
winning_award_2003 = df[(df['Year'] == '2003') & (df['Work'] == 'Road to Perdition') & (df['Result'] == 'Won')]
# Get the award name
award_name = winning_award_2003['Award'].values[0]
print(f"Final Answer: {award_name}")