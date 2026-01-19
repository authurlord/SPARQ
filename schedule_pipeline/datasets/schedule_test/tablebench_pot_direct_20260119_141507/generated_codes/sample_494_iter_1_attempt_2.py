import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Remove the last row which is "Total mainline"
df = df.iloc[:-1]

# Function to parse seats from string
def parse_seats(seat_str):
    if pd.isna(seat_str) or seat_str == '—':
        return np.nan
    if '-' in seat_str:
        parts = seat_str.split('-')
        min_val, max_val = int(parts[0]), int(parts[1])
        return (min_val + max_val) / 2
    else:
        return float(seat_str)

# Apply parsing to the "Seats in Standard Configuration" column
df['avg_seats'] = df['Seats in Standard Configuration'].apply(parse_seats)

# Drop rows where avg_seats is NaN
df = df.dropna(subset=['avg_seats'])

# Find the row with the highest average seat density (i.e., max avg_seats)
max_row = df.loc[df['avg_seats'].idxmax()]

# Return the aircraft type name
final_answer = max_row['Row Header']
print(f"Final Answer: {final_answer}")