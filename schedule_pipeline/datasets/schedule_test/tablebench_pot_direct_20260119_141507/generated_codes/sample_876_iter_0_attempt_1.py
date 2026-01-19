import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'population density 2012 (km 2 )' column
density_values = df['population density 2012 (km 2 )'].astype(float)
max_density = density_values.max()
min_density = density_values.min()
difference = max_density - min_density
print(f"Final Answer: {difference:.1f}")