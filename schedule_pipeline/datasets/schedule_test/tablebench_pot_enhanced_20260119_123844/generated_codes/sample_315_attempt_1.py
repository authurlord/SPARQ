import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Competition is 'World Championships' and Position is 5th or higher
filtered_df = df[df['Competition'] == 'World Championships']
# Define positions that are 5th or better
valid_positions = ['5th (heats)', '5th (semis)', '4th (heats)', '4th (semis)', '3rd', '2nd', '1st']
filtered_df = filtered_df[filtered_df['Position'].isin(valid_positions)]

# Function to convert time string (mm:ss.sss) to seconds
def time_to_seconds(time_str):
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds

# Convert Notes (times) to seconds and find the minimum
times_in_seconds = filtered_df['Notes'].apply(time_to_seconds)
fastest_time_seconds = times_in_seconds.min()

# Convert back to mm:ss.sss format
minutes = int(fastest_time_seconds // 60)
seconds = fastest_time_seconds % 60
fastest_time_formatted = f"{minutes}:{seconds:05.2f}"

print(f"Final Answer: {fastest_time_formatted}")