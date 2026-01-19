import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the last row (total) and sum the '12 - inch' column
total_12_inch_guns = df.iloc[:-1]['12 - inch'].sum()
print(f"Final Answer: {total_12_inch Guns}")