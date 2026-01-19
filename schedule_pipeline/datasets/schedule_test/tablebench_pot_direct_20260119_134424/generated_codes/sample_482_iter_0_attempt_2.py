import pandas as pd

df = pd.read_csv('table.csv')

# Extract career totals
career_total = df.iloc[-1]
career_gp = career_total['Regular season']
career_pts = career_total['Regular season_3']

# Calculate career average points per game
career_avg_pts_per_game = career_pts / career_gp

# Calculate standard deviation of points per game across all seasons
seasons_data = df.iloc[:-1]  # Exclude the total row
seasons_data['pts_per_game'] = seasons_data['Regular season_3'].astype(float) / seasons_data['Regular season'].astype(float)

# Drop rows with zero GP to avoid division by zero
valid_seasons = seasons_data[seasons_data['Regular season'] != '—']
valid_seasons['pts_per_game'] = valid_seasons['pts_per_game'].astype(float)

# Calculate standard deviation
std_dev = valid_seasons['pts_per_game'].std()

# Define threshold for significant deviation (e.g., 1 standard deviation)
threshold = std_dev

# Find seasons with significant deviation
deviant_seasons = valid_seasons[
    (valid_seasons['pts_per_game'] > career_avg_pts_per_game + threshold) |
    (valid_seasons['pts_per_game'] < career_avg_pts_per_game - threshold)
]

# Extract the seasons
deviant_seasons_list = deviant_seasons['Season'].tolist()
print(f"Final Answer: {', '.join(deviant_seasons_list)}")