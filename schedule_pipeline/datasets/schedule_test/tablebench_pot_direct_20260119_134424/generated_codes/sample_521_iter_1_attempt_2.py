import pandas as pd

df = pd.read_csv('table.csv')
# Filter out summary rows like 'Total' and 'Career Total'
clubs = df[df['Club'] != 'Total']
clubs = clubs[clubs['Club'] != 'Career Total']

# Convert 'Total' and 'Total_1' columns to numeric, coercing errors to NaN
clubs['Total'] = pd.to_numeric(clubs['Total'], errors='coerce')
clubs['Total_1'] = pd.to_numeric(clubs['Total_1'], errors='coerce')

# Drop rows with invalid data
clubs.dropna(subset=['Total', 'Total_1'], inplace=True)

# Calculate goal-to-game ratio
clubs['goal_to_game_ratio'] = clubs['Total_1'] / clubs['Total']

# Find the club with the highest ratio
max_ratio_club = clubs.loc[clubs['goal_to_game_ratio'].idxmax(), 'Club']

print(f"Final Answer: {max_ratio_club}")