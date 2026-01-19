import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop the last row (Total mainline) since it's summary
df = df.drop(df.index[-1])

# Function to parse seat range
def parse_seats(seat_str):
    if pd.isna(seat_str) or seat_str == '—':
        return 0
    try:
        if '-' in seat_str:
            parts = seat_str.split('-')
            return (int(parts[0]) + int(parts[1])) / 2
        else:
            return int(seat_str)
    except:
        return 0

# Calculate total seats and total aircraft per type
df['total_seats'] = df['Seats in Standard Configuration'].apply(parse_seats)
df['total_aircraft'] = df['Owned'].astype(float) + df['Leased'].astype(float)

# Total seats and aircraft per type
df['seat_density'] = df['total_seats'] * df['Total'] / df['total_aircraft']

# Find the row with maximum seat density
max_density_row = df.loc[df['seat_density'].idxmax()]
highest_density_type = max_density_row['Row Header']

print(f"Final Answer: {highest_density_type}")