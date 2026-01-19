import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Drop the last row (Total mainline)
df = df.iloc[:-1]

# Function to parse seat configuration
def parse_seats(seats):
    if pd.isna(seats) or seats == '—':
        return 0
    if isinstance(seats, str) and '-' in seats:
        parts = seats.split('-')
        return (int(parts[0]) + int(parts[1])) / 2
    else:
        return int(seats)

# Apply seat parsing
df['seats_avg'] = df['Seats in Standard Configuration'].apply(parse_seats)

# Convert Owned and Leased to integers, treating '—' as 0
df['owned'] = pd.to_numeric(df['Owned'], errors='coerce')
df['leased'] = pd.to_numeric(df['Leased'], errors='coerce')
df['total_aircraft'] = df['owned'] + df['leased']
df['total_seats'] = df['seats_avg'] * df['total_aircraft']

# Calculate average seat density per aircraft type
df['seat_density'] = df['total_seats'] / df['total_aircraft']

# Find the aircraft type with the highest seat density
max_density_row = df.loc[df['seat_density'].idxmax()]
final_answer = max_density_row['Row Header']

print(f"Final Answer: {final_answer}")