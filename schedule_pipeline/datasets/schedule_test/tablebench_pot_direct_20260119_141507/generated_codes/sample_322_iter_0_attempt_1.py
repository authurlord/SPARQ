import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where competition is African Championships and event is 20 km walk
filtered = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Remove rows with missing or invalid positions (like '-' or '?')
filtered = filtered[~(filtered['Position'].str.contains(r'[\-\?]', na=False))]

# Extract time from Notes column (e.g., "1:22:55 (CR)" -> convert to seconds)
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return float('inf')
    parts = time_str.split(':')
    if len(parts) != 3:
        return float('inf')
    minutes, seconds = int(parts[0]), int(parts[1])
    # The third part may have extra text like "(CR)", so we take only the first two
    total_seconds = minutes * 60 + seconds
    return total_seconds

# Apply the function to get seconds
filtered['time_seconds'] = filtered['Notes'].apply(time_to_seconds)

# Find the row with minimum time
best_time_row = filtered.loc[filtered['time_seconds'].idxmin()]

# Return the year of that row
print(f"Final Answer: {best_time_row['Year']}")