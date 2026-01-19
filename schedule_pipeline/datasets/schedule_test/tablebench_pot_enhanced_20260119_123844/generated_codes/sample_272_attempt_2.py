import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
agricultural_panel_column = 'agricultural panel'
filtered_parties = df[df[agricultural_panel_column] >= 2]

# Sum the total seats for these parties
total_seats = filtered_parties['total'].sum()

# Total seats in the entire table
total_all_seats = 60

# Calculate percentage
percentage = (total_seats / total_all_seats) * 100

print(f"Final Answer: {total_seats}, {percentage:.1f}")