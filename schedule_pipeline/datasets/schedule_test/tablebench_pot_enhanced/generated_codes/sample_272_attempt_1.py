import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
qualified_parties = df[df['agricultural panel'] >= 2]
# Sum their total seats
total_seats_qualified = qualified_parties['total'].sum()
# Total seats is 60
total_seats = 60
percentage = (total_seats_qualified / total_seats) * 100

print(f"Final Answer: {total_seats_qualified}, {percentage:.1f}")