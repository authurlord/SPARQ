import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_champs_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Function to convert time string (e.g., '1:22:55') to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == '–' or time_str == 'DNF' or time_str == 'DQ':
        return float('inf')
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        return float('inf')

# Apply conversion and find the minimum time
african_champs_20km['time_seconds'] = african_champs_20km['Notes'].apply(time_to_seconds)
best_time_row = african_champs_20km.loc[african_champs_20km['time_seconds'].idxmin()]

# Get the year of the personal best
best_year = best_time_row['Year']
print(f"Final Answer: {best_year}")