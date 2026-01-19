import pandas as pd

df = pd.read_csv('table.csv')

# Extract player names from points and rebounds columns by removing (number)
def extract_player(x):
    if isinstance(x, str):
        return x.split('(')[0].strip()
    return x

# Apply extraction to points and rebounds
df['points_player'] = df['points'].apply(extract_player)
df['rebounds_player'] = df['rebounds'].apply(extract_player)

# Find the year with the highest points
max_points_year = df.loc[df['points'].str.extract(r'\((\d+)\)', expand=False).astype(int).idxmax()]

# Get the player with highest rebounds in that year
highest_rebounds_player = max_points_year['rebounds_player']

print(f"Final Answer: {highest_rebounds_player}")