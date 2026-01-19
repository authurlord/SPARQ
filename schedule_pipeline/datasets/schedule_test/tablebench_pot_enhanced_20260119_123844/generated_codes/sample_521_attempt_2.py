import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'Career Total' row
clubs = df[df['Club'] != 'Career Total']
# Calculate goal-to-game ratio
clubs['goal_to_game_ratio'] = clubs['Total_1'] / clubs['Total']
# Find the club with the highest ratio
max_ratio_club = clubs.loc[clubs['goal_to_game_ratio'].idxmax(), 'Club']
print(f"Final Answer: {max_ratio_club}")