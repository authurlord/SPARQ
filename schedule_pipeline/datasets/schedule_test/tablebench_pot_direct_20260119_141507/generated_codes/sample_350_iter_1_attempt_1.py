import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract numeric value from 'total goals' column, removing any parentheses
def extract_goals(goals):
    # Remove parentheses and extract the number
    match = re.search(r'\d+', goals)
    return int(match.group()) if match else 0

# Apply the function to create a new column for clean total goals
df['total_goals_numeric'] = df['total goals'].apply(extract_goals)

# Count players with more than 3 goals
players_with_more_than_3_goals = df[df['total_goals_numeric'] > 3]['total_goals_numeric'].count()

print(f"Final Answer: {players_with_more_than_3_goals}")