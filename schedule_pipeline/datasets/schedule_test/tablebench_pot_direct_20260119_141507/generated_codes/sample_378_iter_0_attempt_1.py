import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of elections where 'seats' is 75 or more
count_seats_75_plus = df[df['seats'] >= 75].shape[0]
print(f"Final Answer: {count_seats_75_plus}")