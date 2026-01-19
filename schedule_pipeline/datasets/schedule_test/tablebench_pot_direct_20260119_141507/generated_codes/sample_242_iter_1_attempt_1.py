import pandas as pd

df = pd.read_csv('table.csv')

# Filter for forwards and midfielders
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Extract total goals and convert to integers
forward_goals = forwards['total goals'].astype(int)
midfielder_goals = midfielders['total goals'].astype(int)

# Find maximum goals for each group
max_forward_goals = forward_goals.max()
max_midfielder_goals = midfielder_goals.max()

# Calculate the difference
difference = max_forward_goals - max_midfielder_goals

print(f"Final Answer: {difference}")