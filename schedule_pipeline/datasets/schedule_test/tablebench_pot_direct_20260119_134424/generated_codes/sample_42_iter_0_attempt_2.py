import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'wheat'
wheat_data = df[df['commodity'] == 'wheat']
# Select the required columns and convert to numeric
wheat_values = wheat_data[['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']].astype(int)
# Calculate the average
average_wheat = wheat_values.mean().mean()
print(f"Final Answer: {average_wheat:.1f}")