import pandas as pd

df = pd.read_csv('table.csv')
# Extract wheat production values from 2002-03 to 2005-06
wheat_values = df.loc[df['commodity'] == 'wheat', '2002 - 03':'2005 - 06'].values.flatten()
# Since the data is in string format, convert to integers and compute mean
wheat_values = [int(x) for x in wheat_values]
mean_wheat = sum(wheat_values) / len(wheat_values)
print(f"Final Answer: {mean_wheat}")