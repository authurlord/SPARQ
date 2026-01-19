import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'diameter (km)' to numeric
df['diameter (km)'] = pd.to_numeric(df['diameter (km)'])
# Filter craters with diameter > 40 km
large_craters = df[df['diameter (km)'] > 40]
# Calculate total diameter
total_diameter = large_craters['diameter (km)'].sum()
print(f"Final Answer: {total_diameter}")