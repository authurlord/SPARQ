import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the "total" row and filter parties with at least 2 seats in agricultural panel
filtered_parties = df[df['agricultural panel'] >= 2].drop(index=df.index[-1])  # Drop the total row
total_seats_filtered = filtered_parties['total'].sum()
total_seats_all = df.loc[df.index[-1], 'total']  # Get total seats from the last row
percentage = (total_seats_filtered / total_seats_all) * 100

print(f"Final Answer: {total_seats_filtered}, {percentage:.1f}%")