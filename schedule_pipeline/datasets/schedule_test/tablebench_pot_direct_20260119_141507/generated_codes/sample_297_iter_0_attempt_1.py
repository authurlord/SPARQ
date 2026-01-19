import pandas as pd
import re

# Load the data
df = pd.read_csv('table.csv')

# Function to extract player name from string like "albert white (13)"
def extract_player_name(row):
    text = str(row)
    match = re.search(r'^([^(\s]+)', text)
    if match:
        return match.group(1)
    return None

# Create a list to store player, points, rebounds, year
player_data = []

for idx, row in df.iterrows():
    year = row['year']
    points_str = row['points']
    rebounds_str = row['rebounds']
    
    # Extract player name from points
    points_player = extract_player_name(points_str)
    if points_player is None:
        continue
        
    # Extract player name from rebounds
    rebounds_player = extract_player_name(rebounds_str)
    if rebounds_player is None:
        continue
        
    # If both players are the same, record the data
    if points_player == rebounds_player:
        player_data.append({
            'player': points_player,
            'year': year,
            'points': points_str,
            'rebounds': rebounds_str
        })

# Find the player with the highest points in a year, and check if that player has the highest rebounds in that year
max_points = 0
best_player = None
best_year = None

for entry in player_data:
    # Extract numeric points and rebounds
    try:
        points_val = int(re.search(r'\((\d+)\)', entry['points']).group(1))
        rebounds_val = int(re.search(r'\((\d+)\)', entry['rebounds']).group(1))
    except:
        continue
        
    if points_val > max_points:
        max_points = points_val
        best_player = entry['player']
        best_year = entry['year']

# Final answer: the player with highest points and matching highest rebounds in that year
Final Answer: best_player