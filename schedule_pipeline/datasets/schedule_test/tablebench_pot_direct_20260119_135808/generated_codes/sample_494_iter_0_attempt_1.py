import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract average seats from range or single value
def get_avg_seats(seat_str):
    if '-' in seat_str:
        low, high = map(int, seat_str.split('-'))
        return (low + high) / 2
    else:
        return int(seat_str)

# Apply function to get average seats
df['avg_seats'] = df['Seats in Standard Configuration'].apply(get_avg_seats)

# Calculate average seat density: avg_seats / total aircraft
df['seat_density'] = df['avg_seats'] / df['Total']

# Find the aircraft with the highest seat density
max_density_row = df.loc[df['seat_density'].idxmax()]

print(f"Final Answer: {max_density_row['Row Header']}")