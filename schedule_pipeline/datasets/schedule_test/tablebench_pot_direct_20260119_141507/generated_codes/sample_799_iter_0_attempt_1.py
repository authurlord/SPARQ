import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Count number of drivers per team
team_counts = df['team'].value_counts()
# Get teams with more than one driver
teams_with_multiple_drivers = team_counts[team_counts > 1].index

# Filter drivers who belong to such teams
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Extract 'qual 2' times and compute median
qual2_times = filtered_df['qual 2'].astype(float)
median_qual2 = qual2_times.median()

print(f"Final Answer: {median_qual2:.3f}")