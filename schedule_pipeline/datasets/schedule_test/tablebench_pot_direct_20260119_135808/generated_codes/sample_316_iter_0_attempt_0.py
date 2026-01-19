import pandas as pd

df = pd.read_csv('table.csv')

# Extract the number of titles from the 'winner' column
df['num_titles'] = df['winner'].str.extract(r'(\d+)').astype(int)

# Find the team with the maximum number of titles
max_titles_team = df.loc[df['num_titles'].idxmax(), 'winner']
team_name = max_titles_team.split(' (')[0]  # Extract team name

# Find the first year this team won a title
first_win_year = df[df['winner'].str.contains(team_name, na=False)]['year'].min()

# Get the winning coach for that year
winning_coach = df[df['year'] == first_win_year]['winning coach'].iloc[0]

print(f"Final Answer: {team_name}, {winning_coach}")