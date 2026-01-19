import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract the numeric value from the format "player (points)"
def extract_points(x):
    if isinstance(x, str):
        return int(x.split('(')[-1].rstrip(')'))
    return 0

# Apply function to points column to get numeric values
df['points_numeric'] = df['points'].apply(extract_points)

# Find the year with the maximum points
max_points_year = df.loc[df['points_numeric'].idxmax(), 'year']

# Filter data for that year
year_data = df[df['year'] == max_points_year]

# Extract rebounds for that year and find the player with the highest rebounds
# Similar parsing for rebounds
def extract_rebounds(x):
    if isinstance(x, str):
        return int(x.split('(')[-1].rstrip(')'))
    return 0

year_data['rebounds_numeric'] = year_data['rebounds'].apply(extract_rebounds)
top_rebounder = year_data.loc[year_data['rebounds_numeric'].idxmax(), 'rebounds']

# Extract player name from the rebounds string
player_name = top_rebounder.split(' (')[0]

print(f"Final Answer: {player_name}")