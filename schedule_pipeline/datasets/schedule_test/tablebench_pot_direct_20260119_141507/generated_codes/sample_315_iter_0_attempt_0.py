import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where Competition is 'World Championships' and Position is 5th or higher
# Position values like '5th (semis)' or '4th' are strings, so we check if position starts with '5th' or '4th', etc.
# We consider any position that is 5th or higher (i.e., 5th, 4th, 3rd, 2nd, 1st)
filtered_df = df[(df['Competition'] == 'World Championships') & 
                 (df['Position'].str.contains(r'^(5th|4th|3rd|2nd|1st)', case=False, na=False))]

# Extract the Notes column (contains time like '1:45.61')
times = filtered_df['Notes'].str.extract(r'(\d+:\d+\.\d+)')

# Convert time string to seconds
def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

# Apply conversion and get the minimum time
if not times.empty:
    times_seconds = [time_to_seconds(t) for t in times[0]]
    fastest_time_seconds = min(times_seconds)
    # Convert back to mm:ss format for readability
    m, s = divmod(fastest_time_seconds, 60)
    fastest_time_formatted = f"{int(m):02d}:{s:.2f}"
    print(f"Final Answer: {fastest_time_formatted}")
else:
    print("Final Answer: No valid data found")