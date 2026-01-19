import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Function to compute average seats from a range string
def avg_seats(seat_str):
    if pd.isna(seat_str) or seat_str == '—':
        return 0
    if '-' in seat_str:
        parts = seat_str.split('-')
        return (int(parts[0]) + int(parts[1])) / 2
    else:
        return int(seat_str)

# Apply the function to compute average seats per aircraft
df['avg_seats'] = df['Seats in Standard Configuration'].apply(avg_seats)

# Calculate total seats (average seats * total aircraft)
df['total_seats'] = df['Total'] * df['avg_seats']

# Calculate average seat density (total seats / total aircraft)
df['seat_density'] = df['total_seats'] / df['Total']

# Exclude rows where Total is 0 or missing
df = df[df['Total'] != 0]

# Find the row with maximum seat density
max_density_row = df.loc[df['seat_density'].idxmax()]

# Output the aircraft type with the highest average seat density
print(f"Final Answer: {max_density_row['Row Header']}")