import pandas as pd

df = pd.read_csv('table.csv')

# Step 1: Find teams with more than one driver
team_counts = df['team'].value_counts()
teams_with_multiple_drivers = team_counts[team_counts > 1].index

# Step 2: Filter drivers from teams with more than one driver
filtered_df = df[df['team'].isin(teams_with_multiple_drivers)]

# Step 3: Convert 'qual 2' times to seconds
def time_to_seconds(time_str):
    if ':' in time_str:
        minutes, seconds = map(float, time_str.split(':'))
        return minutes * 60 + seconds
    else:
        return float(time_str)

filtered_df['qual 2 seconds'] = filtered_df['qual 2'].apply(time_to_seconds)

# Step 4: Calculate median
median_qual2 = filtered_df['qual 2 seconds'].median()

print(f"Final Answer: {median_qual2:.3f}")