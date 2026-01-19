import pandas as pd

df = pd.read_csv('table.csv')

# Count occurrences of each team
team_counts = df['team'].value_counts()

# Teams with more than one driver
multi_driver_teams = team_counts[team_counts > 1].index

# Filter drivers from teams with more than one driver
filtered_df = df[df['team'].isin(multi_driver_teams)]

# Function to convert time string to seconds
def time_to_seconds(time_str):
    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

# Convert 'qual 2' to seconds
filtered_df['qual 2 seconds'] = filtered_df['qual 2'].apply(time_to_seconds)

# Calculate median
median_qual2 = filtered_df['qual 2 seconds'].median()

print(f"Final Answer: {median_qual2:.3f}")