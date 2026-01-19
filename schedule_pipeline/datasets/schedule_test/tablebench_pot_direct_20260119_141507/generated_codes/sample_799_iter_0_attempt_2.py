import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Step 1: Count drivers per team
team_count = df['team'].value_counts()
# Step 2: Filter teams with more than one driver
teams_with_multiple_drivers = team_count[team_count > 1].index

# Step 3: Filter rows where team is in those teams
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Step 4: Extract 'qual 2' times and compute median
qual2_times = filtered_df['qual 2'].astype(float)
median_qual2 = qual2_times.median()

print(f"Final Answer: {median_qual2:.3f}")