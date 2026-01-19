import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for Liverpool in the 1995–96 season
row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]

# Extract total goals and total appearances
total_goals = row['Total_1'].iloc[0]  # Goals in the 'Total' column
total_apps = row['Total'].iloc[0]    # Appearances in the 'Total' column

# Calculate goal percentage
goal_percentage = total_goals / total_apps

print(f"Final Answer: {goal_percentage:.3f}")