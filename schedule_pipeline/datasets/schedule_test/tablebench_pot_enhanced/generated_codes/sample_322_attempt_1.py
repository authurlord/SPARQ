import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_championships_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Function to convert time string (e.g., '1:22:55') to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == 'DNF' or time_str == 'DQ' or time_str == '?':
        return float('inf')
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Apply conversion and find the minimum time
african_championships_20km['seconds'] = african_championships_20km['Notes'].apply(time_to_seconds)
best_time_row = african_championships_20km.loc[african_championships_20km['seconds'].idxmin()]

# Extract the year
best_year = best_time_row['Year']
print(f"Final Answer: {best_year}")