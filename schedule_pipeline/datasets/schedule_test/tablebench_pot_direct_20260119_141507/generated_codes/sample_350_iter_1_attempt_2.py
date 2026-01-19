import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract numeric value from 'total goals' column, removing parentheses
def extract_goals(goals_str):
    # Use regex to extract the number before the parenthesis or just the number
    match = re.search(r'(\d+)', goals_str)
    return int(match.group(1)) if match else 0

# Apply the function to the 'total goals' column
df['total_goals_numeric'] = df['total goals'].apply(extract_goals)

# Count players with more than 3 goals
players_with_more_than_3_goals = df[df['total_goals_numeric'] > 3].shape[0]

print(f"Final Answer: {players_with_more_than_3_goals}")