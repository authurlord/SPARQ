import pandas as pd

df = pd.read_csv('table.csv')

# Count how many drivers are in each team
team_counts = df['team'].value_counts()

# Find teams with more than one driver
teams_with_multiple_drivers = team_counts[team_counts > 1].index

# Filter drivers from teams with more than one driver
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Function to convert time string (MM:SS or M:SS) to seconds
def time_to_seconds(time_str):
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # Handle cases like 1:01.040
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)

# Convert 'qual 2' to seconds
filtered_df['qual 2 seconds'] = filtered_df['qual 2'].apply(time_to_seconds)

# Calculate median
median_qual2_seconds = filtered_df['qual 2 seconds'].median()

# Convert back to MM:SS format for readability (though final answer is in seconds)
print(f"Final Answer: {median_qual2_seconds:.3f}")