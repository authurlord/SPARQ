import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'of seats won' column to get total seats won across all elections
total_seats_won = df['of seats won'].sum()
print(f"Final Answer: {total_seats_won}")