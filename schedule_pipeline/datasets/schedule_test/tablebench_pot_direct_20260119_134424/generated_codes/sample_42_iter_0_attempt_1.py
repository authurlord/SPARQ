import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'wheat'
wheat_row = df[df['commodity'] == 'wheat']
# Select the relevant columns
wheat_values = wheat_row[['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']].values[0]
# Convert to integers and calculate mean
average_wheat = sum(int(val) for val in wheat_values) / len(wheat_values)
print(f"Final Answer: {average_wheat:.1f}")