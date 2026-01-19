import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Liverpool in Premier League
liverpool_premier = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]

# Extract the 'Goals' column from 'League_2' (Premier League goals)
liverpool_goals = liverpool_premier['Goals']  # This is the "Goals" under "League_2"

# Find the season with the highest goals
max_goals_row = liverpool_premier.loc[liverpool_goals.idxmax()]
season_with_max_goals = max_goals_row['Season']

print(f"Final Answer: {season_with_max_goals}")