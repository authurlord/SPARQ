import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_champs_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Function to convert time string to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == 'DNF' or time_str == 'DQ' or time_str == '-':
        return float('inf')  # Treat DNF, DQ, – as worst possible
    # Remove any non-numeric characters except colon and parentheses
    clean_time = ''.join(c for c in time_str if c.isdigit() or c == ':')
    if ':' in clean_time:
        parts = clean_time.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float('inf')

# Apply conversion
african_champs_20km['time_seconds'] = african_champs_20km['Notes'].apply(time_to_seconds)

# Find the row with minimum time (best performance)
best_row = african_champs_20km.loc[african_champs_20km['time_seconds'].idxmin()]

# Extract the year
best_year = best_row['Year']
print(f"Final Answer: {best_year}")