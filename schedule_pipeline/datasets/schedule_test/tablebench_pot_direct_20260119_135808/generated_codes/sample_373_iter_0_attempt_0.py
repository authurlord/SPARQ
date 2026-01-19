import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Club is Liverpool
liverpool_data = df[df['Club'] == 'Liverpool']
# Count the number of seasons where League is 'Premier League'
premier_league_seasons = liverpool_data[liverpool_data['League'] == 'Premier League'].shape[0]
print(f"Final Answer: {premier_league_seasons}")