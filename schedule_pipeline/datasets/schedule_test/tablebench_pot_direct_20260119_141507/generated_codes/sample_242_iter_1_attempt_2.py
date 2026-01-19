import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to extract numeric value from string like "3 (1)"
def extract_goals(goals_str):
    match = re.search(r'^(\d+)', goals_str)
    return int(match.group(1)) if match else 0

# Filter players by position
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Extract total goals for forwards and midfielders
forward_goals = forwards['total goals'].apply(extract_goals)
midfielder_goals = midfielders['total goals'].apply(extract_goals)

# Get the maximum goals for each group
top_forward_goals = forward_goals.max()
top_midfielder_goals = midfielder_goals.max()

# Calculate the difference
difference = top_forward_goals - top_midfielder_goals

print(f"Final Answer: {difference}")