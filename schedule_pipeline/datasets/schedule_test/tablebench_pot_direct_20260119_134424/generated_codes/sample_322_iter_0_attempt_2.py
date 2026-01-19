import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_champs_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Convert time to seconds for comparison
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == 'DNF' or time_str == 'DQ':
        return float('inf')
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

african_champs_20km['time_seconds'] = african_champs_20km['Notes'].apply(time_to_seconds)

# Find the row with the minimum time (personal best)
best_time_row = african_champs_20km.loc[african_champs_20km['time_seconds'].idxmin()]

# Extract the year
best_year = best_time_row['Year']
print(f"Final Answer: {best_year}")