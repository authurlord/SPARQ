import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter out rows with "—" in "Seats in Standard Configuration"
df_filtered = df[df['Seats in Standard Configuration'] != '—']

# Convert "Seats in Standard Configuration" to numeric
def parse_seats(seats):
    if pd.isna(seats) or seats == '':
        return 0
    if isinstance(seats, str) and '-' in seats:
        # Parse range: e.g., "267-269" -> average = (267+269)/2 = 268
        a, b = map(int, seats.split('-'))
        return (a + b) / 2
    else:
        return int(seats)

df_filtered['seats_avg'] = df_filtered['Seats in Standard Configuration'].apply(parse_seats)

# Now calculate total seats per aircraft type
df_filtered['total_seats'] = df_filtered['Total'] * df_filtered['seats_avg']

# Calculate average seat density: total seats / total aircraft
df_filtered['seat_density'] = df_filtered['total_seats'] / df_filtered['Total']

# Find the aircraft type with the highest seat density
max_density_row = df_filtered.loc[df_filtered['seat_density'].idxmax()]
highest_density_aircraft = max_density_row['Row Header']

print(f"Final Answer: {highest_density_aircraft}")