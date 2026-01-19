import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where class is 'gt1' and co-drivers contain 'Peter Kox'
filtered_teams = df[(df['class'] == 'gt1') & (df['co - drivers'].str.contains('Peter Kox', na=False))]
# Get the team name
team_name = filtered_teams['team'].values[0] if not filtered_teams.empty else None
print(f"Final Answer: {team_name}")