import pandas as pd

df = pd.read_csv('table.csv')
# Filter row for Liverpool in 1995–96 season
row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]

# Extract total goals and total appearances
total_goals = row['Goals'].iloc[0]
total_apps = row['Total'].iloc[0]

# Calculate goal percentage
goal_percentage = total_goals / total_apps
print(f"Final Answer: {goal_percentage:.3f}")