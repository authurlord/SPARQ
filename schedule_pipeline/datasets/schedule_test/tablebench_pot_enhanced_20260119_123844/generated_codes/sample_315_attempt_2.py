import pandas as pd

df = pd.read_csv('table.csv')

# Filter for World Championships
world_champs = df[df['Competition'] == 'World Championships']

# Filter for positions that are 5th or better (5th, 4th, 3rd, 2nd, 1st)
# We'll check if 'Position' contains any of these
better_than_5th = world_champs[
    world_champs['Position'].str.contains('5th|4th|3rd|2nd|1st', na=False)
]

# Extract the times (Notes column)
times = better_than_5th['Notes']

# Function to convert mm:ss.sss to seconds
def time_to_seconds(t):
    if pd.isna(t):
        return float('inf')
    parts = t.split(':')
    if len(parts) == 2:
        mins = int(parts[0])
        secs = float(parts[1])
        return mins * 60 + secs
    return float('inf')

# Convert all times to seconds and find the minimum
seconds_list = times.apply(time_to_seconds)
fastest_time_seconds = seconds_list.min()

# Convert back to mm:ss.sss format
mins = int(fastest_time_seconds // 60)
secs = fastest_time_seconds % 60
fastest_time_formatted = f"{mins}:{secs:05.2f}"

print(f"Final Answer: {fastest_time_formatted}")