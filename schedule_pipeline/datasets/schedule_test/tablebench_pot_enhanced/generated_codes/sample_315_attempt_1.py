import pandas as pd

df = pd.read_csv('table.csv')

# Filter for World Championships and positions of 5th or higher
filtered_df = df[(df['Competition'] == 'World Championships') & 
                 (df['Position'].isin(['5th (semis)', '4th (heats)', '3rd']))]

# Convert 'Notes' (time) to seconds
def time_to_seconds(time_str):
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds

# Apply conversion
filtered_df['time_seconds'] = filtered_df['Notes'].apply(time_to_seconds)

# Find the minimum time
fastest_time_seconds = filtered_df['time_seconds'].min()

# Convert back to mm:ss format
minutes = int(fastest_time_seconds // 60)
seconds = fastest_time_seconds % 60

# Format to string
fastest_time_formatted = f"{minutes}:{seconds:05.2f}"

print(f"Final Answer: {fastest_time_formatted}")