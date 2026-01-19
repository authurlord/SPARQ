import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'League' is 'Premier League' and the club is 'Liverpool'
premier_league_liverpool = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Count the number of such seasons
season_count = len(premier_league_liverpool)
print(f"Final Answer: {season_count}")