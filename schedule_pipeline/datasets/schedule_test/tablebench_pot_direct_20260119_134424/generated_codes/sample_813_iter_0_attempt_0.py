import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract numeric value from string like '18 (Mensheviks)'
def extract_seats(seat_str):
    try:
        return int(seat_str.split()[0])
    except:
        return 0

# Extract seats for First Duma and Fourth Duma
first_duma_seats = df['First Duma'].apply(extract_seats)
fourth_duma_seats = df['Fourth Duma'].apply(extract_seats)

# Calculate percentage increase
percentage_increase = ((fourth_duma_seats - first_duma_seats) / first_duma_seats) * 100

# Find the party with the highest percentage increase
max_increase_idx = percentage_increase.idxmax()
party_with_max_increase = df.loc[max_increase_idx, 'Party']

print(f"Final Answer: {party_with_max_increase}")