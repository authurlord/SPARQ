import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'wheat'
wheat_row = df[df['commodity'] == 'wheat']
# Select the required columns and convert to integers
values = wheat_row[['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']].values[0]
average_wheat = sum(int(val) for val in values) / len(values)
print(f"Final Answer: {average_wheat:.1f}")