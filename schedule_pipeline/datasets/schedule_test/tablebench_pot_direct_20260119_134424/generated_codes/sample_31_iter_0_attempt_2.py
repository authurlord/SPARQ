import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'of seats won' column
total_seats_won = df['of seats won'].sum()
print(f"Final Answer: {total_seats_won}")