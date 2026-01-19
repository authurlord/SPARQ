import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'wheat'
wheat_row = df[df['commodity'] == 'wheat']
# Select the required columns and convert to integers
wheat_values = wheat_row[['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']].astype(int).values[0]
# Calculate the average
average_wheat = wheat_values.mean()
print(f"Final Answer: {average_wheat:.1f}")