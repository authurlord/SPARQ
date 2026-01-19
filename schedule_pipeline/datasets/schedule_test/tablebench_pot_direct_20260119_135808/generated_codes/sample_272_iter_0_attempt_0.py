import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties with at least 2 seats in the agricultural panel
agricultural_panel = df['agricultural panel'].astype(int)
parties_with_min_2 = df[agricultural_panel >= 2]
total_seats = parties_with_min_2['total'].astype(int).sum()
percentage = (total_seats / 60) * 100
print(f"Final Answer: {total_seats}, {percentage:.1f}")