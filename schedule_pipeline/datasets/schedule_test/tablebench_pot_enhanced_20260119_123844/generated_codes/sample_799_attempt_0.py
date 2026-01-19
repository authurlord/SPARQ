import pandas as pd

df = pd.read_csv('table.csv')

# Count occurrences of each team
team_counts = df['team'].value_counts()

# Teams with more than one driver
multi_driver_teams = team_counts[team_counts > 1].index

# Filter drivers from teams with more than one driver
filtered_df = df[df['team'].isin(multi_driver_teams)]

# Function to convert time string (e.g., '1:01.040') to seconds
def time_to_seconds(time_str):
    if ':' in time_str:
        mins, secs = time_str.split(':')
        return int(mins) * 60 + float(secs)
    else:
        return float(time_str)

# Convert 'qual 2' to seconds
filtered_df['qual 2 seconds'] = filtered_df['qual 2'].apply(time_to_seconds)

# Calculate median
median_qual2 = filtered_df['qual 2 seconds'].median()

# Convert back to mm:ss.ss format for readability
minutes = int(median_qual2 // 60)
seconds = median_qual2 % 60
formatted_median = f"{minutes}:{seconds:05.2f}"

print(f"Final Answer: {formatted_median}")