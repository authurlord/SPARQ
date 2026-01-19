import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool and exclude the total row
liverpool_data = df[df['Club'] == 'Liverpool']
liverpool_data = liverpool_data[liverpool_data['Season'] != 'Liverpool Total']

# Calculate total goals in Premier League (League_1 + League_2)
liverpool_data['Total_Premier_League_Goals'] = liverpool_data['League_1'] + liverpool_data['League_2']

# Find the season with the highest total goals
max_goals_season = liverpool_data.loc[liverpool_data['Total_Premier_League_Goals'].idxmax(), 'Season']

print(f"Final Answer: {max_goals_season}")