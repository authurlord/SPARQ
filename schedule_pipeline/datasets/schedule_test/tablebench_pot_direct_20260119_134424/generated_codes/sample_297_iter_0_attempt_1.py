import pandas as pd

df = pd.read_csv('table.csv')

# Extract the player name and value from 'points' and 'rebounds' columns
def extract_player_value(cell):
    if '(' in cell:
        player = cell.split('(')[0].strip()
        value = int(cell.split('(')[1].split(')')[0])
        return player, value
    return None, None

# Create lists to store max points and corresponding rebound leader
max_points = -1
best_year = None
best_player = None

for index, row in df.iterrows():
    year = row['year']
    points_player, points_value = extract_player_value(row['points'])
    rebounds_player, rebounds_value = extract_player_value(row['rebounds'])
    
    if points_value > max_points:
        max_points = points_value
        best_year = year
        best_player = rebounds_player

print(f"Final Answer: {best_player}")