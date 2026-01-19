import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract player name and value from string like "player (value)"
def extract_player_and_value(x):
    if '(' in x:
        player = x.split('(')[0].strip()
        value = int(x.split('(')[1].split(')')[0])
        return player, value
    return None, None

# Extract points and rebounds data for each year
max_points_year = None
max_points_value = -1
best_rebounder_in_max_points_year = None

for index, row in df.iterrows():
    year = row['year']
    points_player, points_value = extract_player_and_value(row['points'])
    rebounds_player, rebounds_value = extract_player_and_value(row['rebounds'])

    if points_value > max_points_value:
        max_points_value = points_value
        max_points_year = year
        best_rebounder_in_max_points_year = rebounds_player

print(f"Final Answer: {best_rebounder_in_max_points_year}")