import pandas as pd

df = pd.read_csv('table.csv')
# Remove the "total" row (last row) and filter parties with at least 2 seats in agricultural panel
filtered_parties = df[df['agricultural panel'] >= 2].drop(index=df.index[-1])
# Sum the total seats for these parties
total_seats_filtered = filtered_parties['total'].sum()
# Total seats from the last row (total)
total_seats_all = df.iloc[-1]['total']
# Calculate percentage
percentage = (total_seats_filtered / total_seats_all) * 100
print(f"Final Answer: {total_seats_filtered}, {percentage:.1f}%")