import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool and exclude the total row
liverpool_data = df[df['Club'] == 'Liverpool']
# Extract the season with the highest goals in Premier League (League_1)
max_goals_season = liverpool_data.loc[liverpool_data['League_1'].idxmax(), 'Season']
print(f"Final Answer: {max_goals_season}")