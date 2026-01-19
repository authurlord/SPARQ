import pandas as pd

df = pd.read_csv('table.csv')
# Filter team with 13 points and goal difference of 17
filtered_team = df[(df['points'] == 13) & (df['difference'] == 17)]
team_name = filtered_team['team'].values[0] if not filtered_team.empty else None
print(f"Final Answer: {team_name}")