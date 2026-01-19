import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the highest total passengers and highest annual change
max_passengers = df['total passengers'].max()
max_change = df['annual change'].max()

# Filter the row where both are at their peak
unusual_row = df[(df['total passengers'] == max_passengers) & (df['annual change'] == max_change)]
city = unusual_row['location'].values[0]
print(f"Final Answer: {city}")