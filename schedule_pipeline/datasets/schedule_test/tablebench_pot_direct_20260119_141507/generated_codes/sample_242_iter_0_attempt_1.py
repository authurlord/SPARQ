import pandas as pd

df = pd.read_csv('table.csv')
# Filter players by position
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Find the maximum total goals for forwards and midfielders
top_forward_goals = forwards['total goals'].max()
top_midfielder_goals = midfielders['total goals'].max()

# Calculate the difference
difference = top_forward_goals - top_midfielder_goals
print(f"Final Answer: {difference}")