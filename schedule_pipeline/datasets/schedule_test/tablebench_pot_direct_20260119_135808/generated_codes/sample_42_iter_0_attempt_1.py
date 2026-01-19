import pandas as pd

df = pd.read_csv('table.csv')
# Filter for wheat
wheat_data = df[df['commodity'] == 'wheat']
# Select the required years and convert to integers
years = ['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']
wheat_values = wheat_data[years].astype(int).values.flatten()
# Calculate average
average_wheat = wheat_values.mean()
print(f"Final Answer: {average_wheat:.1f}")