import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the highest total passengers and annual change
# The data shows that rank 10 has an unusually high total passengers (100,000,000) and annual change (1000.00%)
unusual_row = df[df['total passengers'] == '100000000']
city = unusual_row['location'].values[0]
print(f"Final Answer: {city}")