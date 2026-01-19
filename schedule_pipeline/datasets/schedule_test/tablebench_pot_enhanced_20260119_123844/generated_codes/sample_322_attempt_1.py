import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_championships_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Function to convert time in format MM:SS or HH:MM:SS to seconds
def time_to_seconds(time_str):
    if ':' not in time_str:
        return float('inf')
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float('inf')

# Apply the function to convert Notes to seconds
african_championships_20km['time_seconds'] = african_championships_20km['Notes'].apply(time_to_seconds)

# Find the row with the minimum time
best_time_row = african_championships_20km.loc[african_championships_20km['time_seconds'].idxmin()]

# Get the year
best_year = best_time_row['Year']
print(f"Final Answer: {best_year}")