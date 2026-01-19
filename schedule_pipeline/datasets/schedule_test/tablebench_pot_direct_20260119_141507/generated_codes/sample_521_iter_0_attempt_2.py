import pandas as pd

df = pd.read_csv('table.csv')

# Filter out the "Career Total" row and keep only club rows
club_rows = df[df['Club'] != 'Career Total']

# Extract total goals and total apps for each club from the 'Total' row
# 'Total' column in the data is at index 11 (Total_1), and 'Goals' is at index 12
# But we need to ensure we pick the correct row: the 'Total' row per club

# The 'Total' row for each club is marked by 'Total' in the Season column
total_rows = df[df['Season'] == 'Total']

# Remove the 'Career Total' row
total_rows = total_rows[total_rows['Club'] != 'Career Total']

# Extract goals and apps for each club
goals = total_rows['Total_1'].astype(float)  # Goals column is at index 12
apps = total_rows['Total'].astype(float)    # Apps column is at index 11

# Compute goal-to-game ratio
ratio = goals / apps

# Find the club with the highest ratio
max_ratio_club = total_rows.iloc[ratio.idxmax()]['Club']
print(f"Final Answer: {max_ratio_club}")