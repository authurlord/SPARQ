import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract numeric value from string like 'player (number)'
def extract_number(value):
    if isinstance(value, str):
        return int(value.split('(')[-1].rstrip(')'))
    return value

# Apply extraction to points and rebounds
df['points_num'] = df['points'].apply(extract_number)
df['rebounds_num'] = df['rebounds'].apply(extract_number)

# Find the year with the highest points
max_points_year = df.loc[df['points_num'].idxmax()]

# Get the player with the highest rebounds in that year
max_rebounds_player = max_points_year['rebounds'].split('(')[0].strip()

print(f"Final Answer: {max_rebounds_player}")