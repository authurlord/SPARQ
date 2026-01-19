import pandas as pd

df = pd.read_csv('table.csv')

# Count how many drivers are in each team
team_counts = df['team'].value_counts()

# Find teams with more than one driver
teams_with_multiple_drivers = team_counts[team_counts > 1].index

# Filter drivers from teams with more than one driver
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Convert 'qual 2' to seconds for proper numeric comparison
def time_to_seconds(time_str):
    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

filtered_df['qual 2'] = filtered_df['qual 2'].apply(time_to_seconds)

# Calculate median
median_qual2 = filtered_df['qual 2'].median()

print(f"Final Answer: {median_qual2:.3f}")