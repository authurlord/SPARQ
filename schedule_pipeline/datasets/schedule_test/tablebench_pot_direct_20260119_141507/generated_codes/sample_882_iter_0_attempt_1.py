import pandas as pd

df = pd.read_csv('table.csv')
# Filter Liverpool's Premier League seasons
liverpool_premier = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]

# Extract the season and goals (from League_2 column, which has Goals)
goals = liverpool_premier['League_2'].astype(float)
seasons = liverpool_premier['Season']

# Find the season with maximum goals
max_goals_idx = goals.idxmax()
max_season = seasons.iloc[goals.idxmax()]

print(f"Final Answer: {max_season}")