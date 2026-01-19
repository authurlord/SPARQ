import pandas as pd

df = pd.read_csv('table.csv')
# Extract wheat production values for the specified years
wheat_values = df.loc[df['commodity'] == 'wheat', '2002 - 03':'2005 - 06'].values.flatten()
# Convert to numeric and calculate mean
mean_wheat = wheat_values.mean()
print(f"Final Answer: {mean_wheat:.1f}")