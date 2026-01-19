import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter the row for Liverpool in the 1995–1996 season
filtered_row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]

# Extract goals and total appearances (from 'Goals' and 'Apps' columns)
goals = filtered_row['Goals'].iloc[0]
apps = filtered_row['Apps'].iloc[0]

# Calculate goal percentage
goal_percentage = (goals / apps) * 100

print(f"Final Answer: {goal_percentage:.2f}")