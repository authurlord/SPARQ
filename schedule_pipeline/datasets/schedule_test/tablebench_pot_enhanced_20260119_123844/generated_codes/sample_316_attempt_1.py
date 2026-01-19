import pandas as pd

df = pd.read_csv('table.csv')

# Extract team name and number of titles from the 'winner' column
df['team_name'] = df['winner'].str.extract(r'([a-zA-Z\s]+)\s*\(')
df['num_titles'] = df['winner'].str.extract(r'\((\d+)\)')

# Convert num_titles to integer
df['num_titles'] = pd.to_numeric(df['num_titles'])

# Find the team with the maximum number of titles
max_titles_team = df.loc[df['num_titles'].idxmax()]

# Get the team name and winning coach for their first title
team_name = max_titles_team['team_name'].strip()
coach_name = max_titles_team['winning coach']

print(f"Final Answer: {team_name}, {coach_name}")