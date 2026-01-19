import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Club is 'Liverpool' and League is 'Premier League'
liverpool_premier_league_goals = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]['Goals'].sum()
print(f"Final Answer: {liverpool_premier_league_goals}")