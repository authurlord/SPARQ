import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the earliest two years: 1893 and 1894
earliest_years = df[df['Year'].isin(['1893', '1894'])]
# Calculate the total quantity
total_quantity = earliest_years['Quantity'].sum()
print(f"Final Answer: {total_quantity}")