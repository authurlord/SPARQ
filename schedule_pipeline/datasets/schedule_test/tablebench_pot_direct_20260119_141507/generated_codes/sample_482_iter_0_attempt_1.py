import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out the "Liiga totals" row which contains overall league stats
liiga_totals = df[df['Season'] == 'Liiga totals'].iloc[0]
regular_season_stats = df[df['Season'] != 'Liiga totals']

# Extract regular season stats (Goals, Assists, Points)
regular_season_data = regular_season_stats[['GP', 'G', 'A', 'Pts']].copy()

# Extract the Liiga totals for comparison
league_avg_g = liiga_totals['G']
league_avg_a = liiga_totals['A']
league_avg_pts = liiga_totals['Pts']

# Calculate deviation ratios for each season
regular_season_data['dev_g'] = (regular_season_data['G'] - league_avg_g) / league_avg_g
regular_season_data['dev_a'] = (regular_season_data['A'] - league_avg_a) / league_avg_a
regular_season_data['dev_pts'] = (regular_season_data['Pts'] - league_avg_pts) / league_avg_pts

# Identify seasons with significant deviation (>2 or < -2 in any stat)
significant_deviation = regular_season_data[
    (regular_season_data['dev_g'] > 2) | (regular_season_data['dev_g'] < -2) |
    (regular_season_data['dev_a'] > 2) | (regular_season_data['dev_a'] < -2) |
    (regular_season_data['dev_pts'] > 2) | (regular_season_data['dev_pts'] < -2)
]

# Get the seasons with significant deviations
deviant_seasons = significant_deviation['Season'].tolist()

# If no significant deviations found, return None
if not deviant_seasons:
    print("Final Answer: none")
else:
    print(f"Final Answer: {', '.join(deviant_seasons)}")