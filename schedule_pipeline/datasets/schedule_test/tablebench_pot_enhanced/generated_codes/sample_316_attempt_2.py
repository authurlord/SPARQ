import pandas as pd

df = pd.read_csv('table.csv')

# Extract team name and number of titles
df['team'] = df['winner'].str.extract(r'([a-zA-Z\s\-\(\)]+)')
df['titles'] = df['winner'].str.extract(r'\((\d+)\)').astype(int)

# Group by team and sum titles
team_titles = df.groupby('team')['titles'].sum()

# Find the team with the most titles
most_titles_team = team_titles.idxmax()
max_titles = team_titles.max()

# Find the first year this team won a title
first_win_year = df[df['team'] == most_titles_team]['year'].min()

# Get the winning coach for the first win
winning_coach = df[(df['team'] == most_titles_team) & (df['year'] == first_win_year)]['winning coach'].iloc[0]

print(f"Final Answer: {most_titles_team}, {winning_coach}")