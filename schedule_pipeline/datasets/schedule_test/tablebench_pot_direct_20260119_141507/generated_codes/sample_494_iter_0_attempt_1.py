import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Remove rows with "—" in Seats in Standard Configuration or where Total is not a number
df = df[df['Seats in Standard Configuration'] != '—']

# Convert "Seats in Standard Configuration" to numeric, handling ranges
def parse_seats(seats):
    if isinstance(seats, str):
        if '-' in seats:
            parts = seats.split('-')
            return (int(parts[0]) + int(parts[1])) / 2
        else:
            return int(seats)
    return float(seats)

df['avg_seats'] = df['Seats in Standard Configuration'].apply(parse_seats)

# Calculate total seats per type: avg_seats * Total
df['total_seats'] = df['avg_seats'] * df['Total'].astype(int)

# Seat density = total_seats / total_aircraft = avg_seats (since total_aircraft = Total)
# So we just need to find the row with max avg_seats

max_density_type = df.loc[df['avg_seats'].idxmax(), 'Row Header']
print(f"Final Answer: {max_density_type}")