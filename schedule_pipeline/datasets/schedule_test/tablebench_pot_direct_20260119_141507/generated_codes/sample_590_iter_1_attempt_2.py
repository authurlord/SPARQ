import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the 'total' row and sum the '12 - inch' column
total_12_inch = df[df['ship'] != 'total']['12 - inch'].sum()
print(f"Final Answer: {total_12_inch}")