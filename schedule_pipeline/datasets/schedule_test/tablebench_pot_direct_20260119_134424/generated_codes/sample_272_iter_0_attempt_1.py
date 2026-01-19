import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'agricultural panel' >= 2
parties_above_2 = df[df['agricultural panel'] >= 2]
# Sum the 'total' column for these parties
total_seats = parties_above_2['total'].sum()
# Total seats is 60
percentage = (total_seats / 60) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")