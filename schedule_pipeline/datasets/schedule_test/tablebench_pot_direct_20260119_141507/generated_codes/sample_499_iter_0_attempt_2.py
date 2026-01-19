import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Liverpool in 1995–1996 season
filtered_row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract total appearances and goals from the 'Total' column (index 11 and 12)
total_apps = filtered_row['Total'].iloc[0]
total_goals = filtered_row['Total_1'].iloc[0]
# Calculate goal percentage
goal_percentage = total_goals / total_apps
print(f"Final Answer: {goal_percentage:.3f}")