import pandas as pd

df = pd.read_csv('table.csv')

# Filter Liverpool data excluding the total row
liverpool_data = df[df['Club'] == 'Liverpool']

# Focus on Premier League seasons (starting from 1992–93)
premier_league_data = liverpool_data[liverpool_data['Season'].str.contains('1992|1993|1994|1995|1996|1997|1998|1999')]

# Find the season with the highest total goals (Total_1)
max_goals_season = premier_league_data.loc[premier_league_data['Total_1'].idxmax()]['Season']

print(f"Final Answer: {max_goals_season}")