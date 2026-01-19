import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
parties_with_agricultural_seats = df[df['agricultural panel'] >= 2]
# Sum their total seats
total_seats = parties_with_agricultural_seats['total'].sum()
# Calculate percentage
percentage = (total_seats / 60) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")