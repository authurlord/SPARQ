import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row and sum the '12 - inch' column
ship_data = df[df['ship'] != 'total']
total_12_inch_guns = ship_data['12 - inch'].sum()
print(f"Final Answer: {total_12_inch Guns}")