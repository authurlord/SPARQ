import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'diameter (km)' to numeric
df['diameter (km)'] = pd.to_numeric(df['diameter (km)'], errors='coerce')
# Filter craters with diameter > 40 km
filtered_diameters = df[df['diameter (km)'] > 40]['diameter (km)']
# Calculate total diameter
total_diameter = filtered_diameters.sum()
print(f"Final Answer: {total_diameter}")