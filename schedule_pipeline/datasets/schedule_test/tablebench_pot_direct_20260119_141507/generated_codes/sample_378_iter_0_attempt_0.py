import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'seats' is 75 or more and count them
count_seats_75_plus = df[df['seats'] >= 75].shape[0]
print(f"Final Answer: {count_seats_75_plus}")