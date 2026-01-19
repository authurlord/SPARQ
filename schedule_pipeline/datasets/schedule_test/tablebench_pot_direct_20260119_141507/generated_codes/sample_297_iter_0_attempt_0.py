import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to extract numeric value from string like "player (value)"
def extract_value(s):
    match = re.search(r'\((\d+)\)', s)
    return int(match.group(1)) if match else 0

# Create new columns for points and rebounds with extracted values
df['points_numeric'] = df['points'].apply(extract_value)
df['rebounds_numeric'] = df['rebounds'].apply(extract_value)

# For each year, find the player with max points and check if they have max rebounds
results = []
for idx, row in df.iterrows():
    year = row['year']
    points = row['points_numeric']
    rebounds = row['rebounds_numeric']
    
    # Get the player name from points or rebounds (using the original string)
    points_player = row['points'].split('(')[0].strip()
    rebounds_player = row['rebounds'].split('(')[0].strip()
    
    # If the player with max points in that year is the same as the one with max rebounds
    if points == row['points_numeric'] and rebounds == row['rebounds_numeric']:
        # Check if the player names match
        if points_player == rebounds_player:
            results.append(points_player)

# If there's a common player with both max points and max rebounds in a year
# We return the first such player (or all unique ones)
unique_results = list(set(results))
if unique_results:
    print(f"Final Answer: {unique_results[0]}")
else:
    print("Final Answer: none")