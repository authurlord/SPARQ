import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract player name from string like 'player (value)'
def extract_player(s):
    if '(' in s:
        return s.split('(')[0].strip()
    return s

# Function to extract value from string like 'player (value)'
def extract_value(s):
    if '(' in s:
        return int(s.split('(')[1].split(')')[0])
    return None

# Create new columns for points and rebounds values and player names
df['points_player'] = df['points'].apply(extract_player)
df['points_value'] = df['points'].apply(extract_value)
df['rebounds_player'] = df['rebounds'].apply(extract_player)
df['rebounds_value'] = df['rebounds'].apply(extract_value)

# Find the year with the highest points
max_points_year = df.loc[df['points_value'].idxmax()]
max_points_player = max_points_year['points_player']
max_points_value = max_points_year['points_value']

# Check if this player also has the highest rebounds in the same year
if max_points_year['rebounds_value'] == max_points_year['rebounds_value']:
    final_answer = max_points_player
else:
    # If not, find the player with max rebounds in that year
    final_answer = max_points_year['rebounds_player']

print(f"Final Answer: {final_answer}")