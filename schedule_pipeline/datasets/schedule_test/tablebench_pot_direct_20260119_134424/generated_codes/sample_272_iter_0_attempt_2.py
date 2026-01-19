import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
agricultural_panel = df[df['agricultural panel'] >= '2']
# Convert 'total' column to numeric for summation
agricultural_panel['total'] = pd.to_numeric(agricultural_panel['total'])
# Sum total seats for these parties
total_seats = agricultural_panel['total'].sum()
# Calculate percentage
percentage = (total_seats / 60) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")