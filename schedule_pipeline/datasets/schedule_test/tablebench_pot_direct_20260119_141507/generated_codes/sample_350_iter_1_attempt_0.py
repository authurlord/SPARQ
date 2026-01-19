import pandas as pd
import re

df = pd.read_csv('table.csv')

# Parse 'total goals' column: extract numeric value before parentheses
def parse_goals(goals):
    # Remove anything in parentheses and convert to int
    match = re.search(r'^(\d+)', goals)
    if match:
        return int(match.group(1))
    return 0

# Apply the parsing function to the 'total goals' column
df['total_goals_numeric'] = df['total goals'].apply(parse_goals)

# Count players with total goals > 3
count = df[df['total_goals_numeric'] > 3].shape[0]

print(f"Final Answer: {count}")