import pandas as pd
import re

df = pd.read_csv('table.csv')

# Filter relevant rows
filtered = df[
    (df['Competition'] == 'African Championships') &
    (df['Event'] == '20 km walk') &
    (df['Position'] != '-') &
    (df['Position'] != '–')
]

# Function to convert time string to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return float('inf')
    parts = re.split(r':', time_str.strip())
    if len(parts) != 3:
        return float('inf')
    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    return hours * 3600 + minutes * 60 + seconds

# Apply conversion to Notes column
filtered['time_seconds'] = filtered['Notes'].apply(time_to_seconds)

# Find the row with minimum time
best_time_row = filtered.loc[filtered['time_seconds'].idxmin()]

# Extract the year from the first row (Year is in the first column)
final_year = best_time_row.iloc[0]  # First column is Year

print(f"Final Answer: {final_year}")