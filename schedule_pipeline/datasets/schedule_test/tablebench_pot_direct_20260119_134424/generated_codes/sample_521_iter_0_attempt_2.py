import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'Career Total' row to get overall totals
career_total = df[df['Club'] == 'Career Total'].iloc[0]

# Get total goals and total games from Career Total
total_goals = career_total['Total']
total_games = career_total['Total_1']

# Calculate overall goal-to-game ratio
overall_ratio = total_goals / total_games

# Now, find the club with the highest goal-to-game ratio among individual clubs
# We'll consider only the rows where Club is not 'Career Total' and has a 'Total' entry
club_rows = df[df['Club'] != 'Career Total']
club_rows = club_rows[club_rows['Season'] == 'Total']

# Calculate goal-to-game ratio for each club
club_rows['goal_to_game_ratio'] = club_rows['Total'] / club_rows['Total_1']

# Find the club with the highest ratio
max_ratio_club = club_rows.loc[club_rows['goal_to_game_ratio'].idxmax()]['Club']

print(f"Final Answer: {max_ratio_club}")