import pandas as pd

df = pd.read_csv('table.csv')

# Filter out summary rows (Total and Career Total)
clubs_data = df[df['Club'] != 'Total']
clubs_data = clubs_data[clubs_data['Club'] != 'Career Total']

# Calculate goal-to-game ratio
clubs_data['goal_to_game_ratio'] = clubs_data['Total_1'] / clubs_data['Total']

# Find the club with the highest ratio
max_ratio_club = clubs_data.loc[clubs_data['goal_to_game_ratio'].idxmax(), 'Club']

print(f"Final Answer: {max_ratio_club}")