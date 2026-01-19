import pandas as pd

df = pd.read_csv('table.csv')

# Count number of drivers per team
team_counts = df['team'].value_counts()

# Teams with more than one driver
multi_driver_teams = team_counts[team_counts > 1].index

# Filter drivers from teams with more than one driver
filtered_df = df[df['team'].isin(multi_driver_teams)]

# Function to convert time string to seconds
def time_to_seconds(time_str):
    if isinstance(time_str, str) and ':' in time_str:
        m, s = map(float, time_str.split(':'))
        return m * 60 + s
    else:
        return float(time_str)

# Convert 'qual 2' to seconds
filtered_df['qual_2_seconds'] = filtered_df['qual 2'].apply(time_to_seconds)

# Calculate median
median_seconds = filtered_df['qual_2_seconds'].median()

print(f"Final Answer: {median_seconds:.3f}")