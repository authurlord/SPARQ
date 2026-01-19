import pandas as pd

df = pd.read_csv('table.csv')
# Sum the '12 - inch' column
total_12_inch_guns = df['12 - inch'].sum()
print(f"Final Answer: {total_12_inch_guns}")