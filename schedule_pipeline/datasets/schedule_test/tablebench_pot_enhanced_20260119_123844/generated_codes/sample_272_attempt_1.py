import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
agricultural_panel = df['agricultural panel']
parties_with_2_or_more = df[agricultural_panel >= 2]
# Sum total seats for these parties
total_seats = parties_with_2_or_more['total'].sum()
# Calculate percentage
percentage = (total_seats / 60) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")