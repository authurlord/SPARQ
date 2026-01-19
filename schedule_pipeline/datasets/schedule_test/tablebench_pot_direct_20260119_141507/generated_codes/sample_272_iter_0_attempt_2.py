import pandas as pd

df = pd.read_csv('table.csv')

# Filter parties with at least 2 seats in agricultural panel
filtered_parties = df[df['agricultural panel'] >= 2]

# Sum the total seats for these parties
total_seats_selected = filtered_parties['total'].sum()

# Total seats across all parties (from the last row)
total_seats_all = df.iloc[-1]['total']

# Calculate percentage
percentage = (total_seats_selected / total_seats_all) * 100

print(f"Final Answer: {total_seats_selected}, {percentage:.1f}")