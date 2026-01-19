import pandas as pd

df = pd.read_csv('table.csv')
# Extract seats for Kadets in First and Second Duma
first_duma_seats = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['First Duma'].values[0])
second_duma_seats = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['Second Duma'].values[0])

# Calculate seat share change
seat_share_change = ((second_duma_seats - first_duma_seats) / first_duma_seats) * 100

print(f"Final Answer: {seat_share_change:.1f}")