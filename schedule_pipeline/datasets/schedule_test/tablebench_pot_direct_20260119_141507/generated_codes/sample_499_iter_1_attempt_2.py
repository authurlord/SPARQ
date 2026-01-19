import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Liverpool in 1995–96 season
filtered_row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract total goals and total appearances
total_goals = filtered_row['Total_1'].values[0]
total_apps = filtered_row['Total'].values[0]
# Calculate goal percentage
goal_percentage = (total_goals / total_apps) * 100
print(f"Final Answer: {goal_percentage:.2f}")