import pandas as pd

df = pd.read_csv('table.csv')

# Filter for World Championships with 800m and position 5th or higher
filtered = df[
    (df['Competition'] == 'World Championships') &
    (df['Event'] == '800 m') &
    (df['Position'].str.contains(r'^(5th|4th|3rd|2nd|1st)', case=False, na=False))
]

# Extract the time from 'Notes' column (e.g., '1:45.61')
times = []
for note in filtered['Notes']:
    if ':' in note:
        parts = note.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        total_seconds = minutes * 60 + seconds
        times.append(total_seconds)

# Find the minimum (fastest) time
fastest_time_seconds = min(times)
fastest_time_formatted = f"{int(fastest_time_seconds // 60):01d}:{fastest_time_seconds % 60:.2f}"

print(f"Final Answer: {fastest_time_formatted}")