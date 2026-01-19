import pandas as pd

df = pd.read_csv('table.csv')

# Convert agricultural panel to integer
df['agricultural panel'] = df['agricultural panel'].astype(int)

# Filter parties with at least 2 seats in agricultural panel
filtered_parties = df[df['agricultural panel'] >= 2]

# Calculate total seats for these parties
total_seats_filtered = filtered_parties['total'].sum()

# Total seats across all parties
total_seats_all = df['total'].iloc[-1]  # Last row is total

# Calculate percentage
percentage = (total_seats_filtered / total_seats_all) * 100

print(f"Final Answer: {total_seats_filtered}, {percentage:.1f}%")