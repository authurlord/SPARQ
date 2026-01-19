import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract the seat count (midpoint if range, else use the number)
def get_seat_count(seats_str):
    if '-' in seats_str:
        low, high = map(int, seats_str.split('-'))
        return (low + high) / 2
    else:
        return int(seats_str)

# Apply the function to get average seat count
df['avg_seats'] = df['Seats in Standard Configuration'].apply(get_seat_count)

# Calculate seat density (average seats per aircraft)
df['seat_density'] = df['avg_seats'] / df['Total']

# Exclude the 'Total mainline' row
aircraft_types = df.loc[df['Row Header'] != 'Total mainline', 'Row Header']
seat_densities = df.loc[df['Row Header'] != 'Total mainline', 'seat_density']

# Find the aircraft type with the highest seat density
max_density_aircraft = aircraft_types[seat_densities.idxmax()]

print(f"Final Answer: {max_density_aircraft}")