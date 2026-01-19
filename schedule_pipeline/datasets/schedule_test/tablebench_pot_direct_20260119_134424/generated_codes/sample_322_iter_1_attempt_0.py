import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_champs_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Function to convert time string to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str in ['DNF', 'DQ', '?']:
        return float('inf')  # Skip invalid entries
    # Remove any non-time text like (CR)
    time_clean = ''.join(c for c in time_str if c.isdigit() or c == ':')
    parts = time_clean.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        return float('inf')

# Apply conversion and find the minimum time
african_champs_20km['time_seconds'] = african_champs_20km['Notes'].apply(time_to_seconds)
best_time_row = african_champs_20km.loc[african_champs_20km['time_seconds'].idxmin()]

# Extract the year of the personal best
final_year = best_time_row['Year']
print(f"Final Answer: {final_year}")