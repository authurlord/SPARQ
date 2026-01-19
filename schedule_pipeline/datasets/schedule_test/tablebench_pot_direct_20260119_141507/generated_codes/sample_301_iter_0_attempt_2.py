import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the data to a list of dictionaries for easier manipulation
data = df.to_dict('records')

# Dictionary to store player appearances in mixed doubles when they also won men's singles
player_count = {}

# Iterate over each row (year)
for row in data:
    year = row['year']
    mens_singles = row['mens singles']
    mixed_doubles = row['mixed doubles']
    
    # Split mixed doubles into individual names
    mixed_names = mixed_doubles.split()
    
    # Check if any of the mixed doubles players matches the men's singles winner
    for name in mixed_names:
        if name == mens_singles:
            # If the player is the same, increment count
            if name in player_count:
                player_count[name] += 1
            else:
                player_count[name] = 1

# Find players who have won mixed doubles at least twice in years when they also won men's singles
result = [name for name, count in player_count.items() if count >= 2]

print(f"Final Answer: {', '.join(result)}")