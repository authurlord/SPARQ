import pandas as pd

df = pd.read_csv('table.csv')
# Filter for wheat
wheat_row = df[df['commodity'] == 'wheat']
# Select the required years
years = ['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']
wheat_values = wheat_row[years].values[0]
# Convert to integers and calculate average
average_wheat = sum(int(x) for x in wheat_values) / len(wheat_values)
print(f"Final Answer: {average_wheat:.1f}")