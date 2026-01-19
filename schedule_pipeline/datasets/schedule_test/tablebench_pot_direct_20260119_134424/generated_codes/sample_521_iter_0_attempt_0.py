import pandas as pd

df = pd.read_csv('table.csv')
# Filter out total rows and keep only club data
club_data = df[df['Club'] != 'Total']
club_data = club_data[club_data['Club'] != 'Career Total']

# Calculate goal-to-game ratio
club_data['goal_to_game_ratio'] = club_data['Total_1'] / club_data['Total']

# Find the club with the highest ratio
max_ratio_club = club_data.loc[club_data['goal_to_game_ratio'].idxmax(), 'Club']

print(f"Final Answer: {max_ratio_club}")