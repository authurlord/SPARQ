import pandas as pd

df = pd.read_csv('table.csv')
# Extract wheat production values from 2002-03 to 2005-06
wheat_data = df.loc[df['commodity'] == 'wheat', '2002 - 03':'2005 - 06'].values.flatten()
# Convert to numeric and calculate mean
mean_wheat = wheat_data.mean()
print(f"Final Answer: {mean_wheat:.1f}")