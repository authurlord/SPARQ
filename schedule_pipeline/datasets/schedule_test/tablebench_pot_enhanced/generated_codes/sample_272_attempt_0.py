import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'agricultural panel' >= 2
qualified_parties = df[df['agricultural panel'] >= 2]
# Sum the 'total' seats for these parties
total_seats = qualified_parties['total'].sum()
# Total seats in the system
total_system_seats = 60
# Calculate percentage
percentage = (total_seats / total_system_seats) * 100
print(f"Final Answer: {total_seats}, {percentage:.2f}")