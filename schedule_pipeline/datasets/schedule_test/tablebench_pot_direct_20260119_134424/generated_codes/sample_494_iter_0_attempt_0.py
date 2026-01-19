import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract the average seat count from a range or return the value
def get_avg_seats(seat_str):
    if '-' in seat_str:
        low, high = map(int, seat_str.split('-'))
        return (low + high) / 2
    else:
        return int(seat_str)

# Apply the function to 'Seats in Standard Configuration'
df['avg_seats'] = df['Seats in Standard Configuration'].apply(get_avg_seats)

# Calculate seat density: avg_seats / Total
df['seat_density'] = df['avg_seats'] / df['Total']

# Find the aircraft with the highest seat density
max_density_row = df.loc[df['seat_density'].idxmax()]

print(f"Final Answer: {max_density_row['Row Header']}")