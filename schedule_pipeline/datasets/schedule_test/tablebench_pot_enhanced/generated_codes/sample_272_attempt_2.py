import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
filtered_df = df[df['agricultural panel'] >= 2]
# Sum the total seats for these parties
total_seats = filtered_df['total'].sum()
# Calculate percentage of total seats (60)
percentage = (total_seats / 60) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")