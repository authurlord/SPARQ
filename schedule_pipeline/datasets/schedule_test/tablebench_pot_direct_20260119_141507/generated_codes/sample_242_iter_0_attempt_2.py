import pandas as pd

df = pd.read_csv('table.csv')

# Filter for forwards and midfielders
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Find top scorer in each group by total goals
top_forward_goals = forwards['total goals'].max()
top_midfielder_goals = midfielders['total goals'].max()

# Calculate the difference
difference = top_forward_goals - top_midfielder_goals
print(f"Final Answer: {difference}")