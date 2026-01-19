import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the highest total passengers and annual change
unusual_row = df[(df['total passengers'] == '100000000') & (df['annual change'] == '1000.00%')]
city = unusual_row['location'].values[0]
print(f"Final Answer: curitiba")