import pandas as pd

df = pd.read_csv('table.csv')

# Filter for forwards (fw) and midfielders (mf)
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Find the maximum total goals for each position
top_fw_goals = forwards['total goals'].max()
top_mf_goals = midfielders['total goals'].max()

# Calculate the difference
difference = top_fw_goals - top_mf_goals

print(f"Final Answer: {difference}")