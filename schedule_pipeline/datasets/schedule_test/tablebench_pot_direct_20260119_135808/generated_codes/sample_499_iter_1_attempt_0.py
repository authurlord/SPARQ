import pandas as pd

df = pd.read_csv('table.csv')

# Filter the data for Liverpool in the 1995-1996 season
liverpool_1995_1996 = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]

# Extract goals and appearances from the relevant columns
goals = liverpool_1995_1996['League'].iloc[0]  # Goals from League column
appearances = liverpool_1995_1996['League_1'].iloc[0]  # Appearances from League_1 column

# Calculate goal percentage
goal_percentage = (goals / appearances) * 100 if appearances != 0 else 0

# Print final answer rounded to two decimal places
print(f"Final Answer: {goal_percentage:.2f}")