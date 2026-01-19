import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract numeric value from string like 'player (points)'
def extract_value(x):
    if isinstance(x, str):
        return int(x.split('(')[-1].rstrip(')'))
    return x

# Apply extraction to points and rebounds columns
df['points'] = df['points'].apply(extract_value)
df['rebounds'] = df['rebounds'].apply(extract_value)

# Find the year with the maximum points
max_points_year = df.loc[df['points'].idxmax()]

# Get the player with the highest rebounds in that year
max_rebounds_player = max_points_year['rebounds']

# Extract the player name from the rebounds column (original string format)
rebounds_str = df.iloc[df['points'].idxmax()]['rebounds']
rebounds_player = rebounds_str.split('(')[0].strip()

print(f"Final Answer: {rebounds_player}")