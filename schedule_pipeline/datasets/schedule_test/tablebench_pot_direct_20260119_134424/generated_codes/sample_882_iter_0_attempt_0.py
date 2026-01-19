import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in Premier League seasons
liverpool_premier = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Find the season with the highest total goals
max_goals_season = liverpool_premier.loc[liverpool_premier['Total_1'].idxmax()]['Season']
print(f"Final Answer: {max_goals_season}")