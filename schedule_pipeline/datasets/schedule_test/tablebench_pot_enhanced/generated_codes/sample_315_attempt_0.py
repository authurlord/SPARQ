import pandas as pd

df = pd.read_csv('table.csv')

# Filter for World Championships and positions 5th or higher
world_championships = df[df['Competition'] == 'World Championships']
valid_positions = ['5th (heats)', '5th (semis)', '4th (heats)', '4th (semis)', '3rd', '2nd', '1st', '1st']
filtered_data = world_championships[world_championships['Position'].isin(valid_positions)]

# Ensure event is 800m
filtered_data = filtered_data[filtered_data['Event'].str.contains('800m|800 m', case=False, na=False)]

# Convert time in Notes to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return float('inf')
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds

filtered_data['time_seconds'] = filtered_data['Notes'].apply(time_to_seconds)

# Find the fastest time
fastest_time_seconds = filtered_data['time_seconds'].min()

# Convert back to mm:ss.ss format
minutes = int(fastest_time_seconds // 60)
seconds = fastest_time_seconds % 60
fastest_time_formatted = f"{minutes}:{seconds:05.2f}"

print(f"Final Answer: {fastest_time_formatted}")