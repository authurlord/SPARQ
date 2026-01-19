import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where class is 'gt1' and co-drivers contain 'Peter Kox'
filtered_teams = df[(df['class'] == 'gt1') & (df['co - drivers'].str.contains('Peter Kox', na=False))]['team']
# Return the team name
print(f"Final Answer: {filtered_teams.iloc[0]}")