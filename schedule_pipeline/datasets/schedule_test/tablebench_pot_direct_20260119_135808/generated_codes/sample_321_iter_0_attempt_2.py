import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 'gt1' class and co-drivers containing 'peter kox'
filtered_df = df[(df['class'] == 'gt1') & (df['co-drivers'].str.contains('peter kox', case=False, na=False))]
# Extract the team name
team = filtered_df['team'].values[0]
print(f"Final Answer: {team}")