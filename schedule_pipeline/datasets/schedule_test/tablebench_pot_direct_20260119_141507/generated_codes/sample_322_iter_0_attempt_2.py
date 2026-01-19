import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Competition is 'African Championships' and Event is '20 km walk'
filtered_df = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Remove rows where Notes is NaN or empty
filtered_df = filtered_df.dropna(subset=['Notes'])

# Parse time from Notes (e.g., "1:22:55" -> total seconds)
def parse_time(time_str):
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = int(parts[0]), int(parts[1])
        return minutes * 60 + seconds
    return float('inf')

# Apply parsing to get time in seconds
filtered_df['time_seconds'] = filtered_df['Notes'].apply(parse_time)

# Find the row with minimum time (best performance)
best_performance = filtered_df.loc[filtered_df['time_seconds'].idxmin()]

# Extract the year from the row
year = best_performance['Year']
print(f"Final Answer: {year}")