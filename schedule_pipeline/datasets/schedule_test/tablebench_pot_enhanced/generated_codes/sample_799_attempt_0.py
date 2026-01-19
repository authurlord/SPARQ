import pandas as pd

df = pd.read_csv('table.csv')

# Step 1: Count how many drivers per team
team_counts = df['team'].value_counts()

# Step 2: Find teams with more than one driver
teams_with_multiple_drivers = team_counts[team_counts > 1].index

# Step 3: Filter data for drivers from these teams
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Step 4: Convert 'qual 2' times to seconds
def to_seconds(time_str):
    if ':' in time_str:
        m, s = map(float, time_str.split(':'))
        return m * 60 + s
    else:
        return float(time_str)

filtered_df['qual_2_seconds'] = filtered_df['qual 2'].apply(to_seconds)

# Step 5: Calculate median
median_time_seconds = filtered_df['qual_2_seconds'].median()

# Convert back to mm:ss format for clarity
minutes = int(median_time_seconds // 60)
seconds = median_time_seconds % 60
print(f"Final Answer: {minutes}:{seconds:.2f}")