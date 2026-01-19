import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'agricultural panel' >= 2
parties_above_2 = df[df['agricultural panel'] >= 2]
# Sum the total seats for these parties
total_seats = parties_above_2['total'].sum()
# Calculate percentage
percentage = (total_seats / df['total'].iloc[-1]) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")