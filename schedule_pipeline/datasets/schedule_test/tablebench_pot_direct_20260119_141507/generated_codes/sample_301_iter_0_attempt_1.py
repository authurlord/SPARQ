import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Create a list to store players who meet the criteria
valid_players = []

# Iterate through each year
for idx, row in df.iterrows():
    mens_singles = row['mens singles']
    mixed_doubles = row['mixed doubles']
    
    # Check if the same player won both
    if mens_singles == mixed_doubles:
        valid_players.append(mens_singles)
    elif mens_singles in mixed_doubles:
        # If the player appears in mixed doubles (as a string), we need to parse it
        # But mixed doubles is a list of two names, so we need to check if mens_singles is one of them
        # Split mixed doubles by space and check
        mixed_names = mixed_doubles.split()
        if mens_singles in mixed_names:
            valid_players.append(mens_singles)

# Now count occurrences of each player in valid_players
from collections import Counter
player_counts = Counter(valid_players)

# Find players with at least 2 appearances
result_players = [player for player, count in player_counts.items() if count >= 2]

print(f"Final Answer: {', '.join(result_players)}")