import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for Liverpool in the 1995–96 season
filtered_row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]

# Extract goals and appearances (from the 'Goals' column in the row)
goals = filtered_row.iloc[0]['Goals']  # This is the 5th column in the row
apps = filtered_row.iloc[0]['Apps']   # This is the 4th column in the row

# Calculate goal percentage
goal_percentage = goals / apps

print(f"Final Answer: {goal_percentage:.3f}")