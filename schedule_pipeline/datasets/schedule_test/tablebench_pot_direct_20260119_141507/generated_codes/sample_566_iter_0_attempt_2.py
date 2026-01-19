import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for the earliest two years: 1893 and 1894
filtered_df = df[(df['Year'] == '1893') | (df['Year'] == '1894')]
# Sum the quantity of orders
total_quantity = filtered_df['Quantity'].sum()
print(f"Final Answer: {total_quantity}")