import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'total goals' column by extracting numeric values
def extract_goals(goals_str):
    match = re.search(r'^(\d+)', goals_str)
    return int(match.group(1)) if match else 0

df['total goals'] = df['total goals'].apply(extract_goals)

# Filter for forwards (fw) and midfielders (mf)
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Get the maximum total goals for each group
top_forward_goals = forwards['total goals'].max()
top_midfielder_goals = midfielders['total goals'].max()

# Calculate the difference
difference = top_forward_goals - top_midfielder_goals

print(f"Final Answer: {difference}")