import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the earliest two years: 1893 and 1894
earliest_two_years = df[df['Year'].isin(['1893', '1894'])]
# Sum the Quantity for these rows
total_quantity = earliest_two_years['Quantity'].sum()
print(f"Final Answer: {total_quantity}")