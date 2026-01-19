import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'population density 2012 (km 2 )' to numeric to perform calculations
df['population density 2012 (km 2 )'] = pd.to_numeric(df['population density 2012 (km 2 )'])
# Calculate the difference between highest and lowest population density
density_diff = df['population density 2012 (km 2 )'].max() - df['population density 2012 (km 2 )'].min()
print(f"Final Answer: {density_diff}")