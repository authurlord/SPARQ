import pandas as pd

df = pd.read_csv('table.csv')

# Count number of drivers per team
team_counts = df['team'].value_counts()
# Filter teams with more than one driver
teams_with_multiple_drivers = team_counts[team_counts > 1].index

# Filter rows where team is in those teams
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Extract 'qual 2' times and calculate median
qual2_times = filtered_df['qual 2'].astype(float)
median_qual2 = qual2_times.median()

print(f"Final Answer: {median_qual2:.3f}")