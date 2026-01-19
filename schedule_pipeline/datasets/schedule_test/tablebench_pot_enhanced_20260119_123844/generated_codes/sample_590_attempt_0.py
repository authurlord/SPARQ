import pandas as pd

df = pd.read_csv('table.csv')
# Sum the '12 - inch' column to get total number of 12-inch guns
total_12inch_guns = df['12 - inch'].sum()
print(f"Final Answer: {total_12inch_guns}")