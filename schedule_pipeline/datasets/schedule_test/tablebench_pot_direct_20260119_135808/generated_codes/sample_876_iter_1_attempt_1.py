import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'population density 2012 (km 2 )' to numeric
df['population density 2012 (km 2 )'] = pd.to_numeric(df['population density 2012 (km 2 )'])
# Find max and min population density
max_density = df['population density 2012 (km 2 )'].max()
min_density = df['population density 2012 (km 2 )'].min()
# Calculate the difference
difference = max_density - min_density
print(f"Final Answer: {difference}")