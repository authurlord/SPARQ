import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for Liverpool in the 1995–96 season
filtered_row = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]

# Extract goals and total appearances (from the 'Goals' column in the 'Total' column)
# The last column is 'Total_1', which contains total apps and goals
# But for the season, we need the goals and apps from the relevant columns

# From the row, we can directly get:
# 'Goals' is in the 6th column (index 5) — "Goals" in the "League" section
# 'Apps' is in the 4th column (index 3) — "Apps" in the "League" section

# Actually, looking at the structure:
# The row has: [Club, Season, Division, Apps, Goals, Apps, Goals, Apps, Goals, Apps, Goals, Apps, Goals]
# So:
# Index 3: Apps (League)
# Index 4: Goals (League)
# Index 11: Total Apps (Total)
# Index 12: Total Goals (Total)

# For goal percentage, we want Goals / Total Apps (from the "Total" column)

# Get the row
row = filtered_row.iloc[0]
total_apps = row['Total']  # This is the last column before "Total_1"
total_goals = row['Total_1']  # This is the final "Goals" column

# But wait — the "Total" and "Total_1" are actually the last two columns
# The "Total" column is at index 11, and "Total_1" is at index 12
# So we take:
# Total appearances = row['Total'] (index 11)
# Total goals = row['Total_1'] (index 12)

# However, in the row, the last two entries are:
# '53', '9' → so total apps = 53, total goals = 9

# Therefore:
goal_percentage = row['Total_1'] / row['Total']

print(f"Final Answer: {goal_percentage:.3f}")