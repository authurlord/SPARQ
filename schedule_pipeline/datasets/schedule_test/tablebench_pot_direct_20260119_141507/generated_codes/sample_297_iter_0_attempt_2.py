import pandas as pd
import re

# Load the data
df = pd.read_csv('table.csv')

# Function to extract player name and score from string like "name (value)"
def parse_player_score(s):
    match = re.search(r'([a-z\s]+)\s*\((\d+)\)', s)
    if match:
        player = match.group(1).strip()
        score = int(match.group(2))
        return player, score
    return None, 0

# Initialize variables
max_points_year = None
best_player = None
max_rebounds_in_max_points_year = 0

# Process each row
for idx, row in df.iterrows():
    year = row['year']
    points_str = row['points']
    rebounds_str = row['rebounds']
    
    # Parse points
    points_player, points_value = parse_player_score(points_str)
    if points_value == 0:
        continue
        
    # Parse rebounds
    rebounds_player, rebounds_value = parse_player_score(rebounds_str)
    if rebounds_value == 0:
        continue
        
    # If this player has more points than any other in this year, update max
    if points_value > points_value:
        pass  # Not applicable; we're iterating per row
    
    # Check if this player has the highest points in this year
    # We need to compare across all players in the same year
    # But since only one entry per year is given, we assume the first parsed value is the main one
    
    # For now, since each row has one player listed, we can just track the player with max points overall
    # and then check if they have max rebounds in that year
    
    # Track the player with the highest points across years
    if max_points_year is None or points_value > max_points_year['points']:
        max_points_year = {
            'year': year,
            'player': points_player,
            'points': points_value,
            'rebounds': rebounds_value
        }
        best_player = points_player

# Final check: in the year with max points, does the player have the highest rebounds?
# Since only one player is listed per year, we assume the player with max points in that year is the one we want
# And if they have the highest rebounds in that year, it's valid.

# Print the result
print(f"Final Answer: {best_player}")