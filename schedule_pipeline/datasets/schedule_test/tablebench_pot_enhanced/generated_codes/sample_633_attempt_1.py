import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where diameter > 40 km
filtered_df = df[df['diameter (km)'] > 40]
# Calculate total diameter
total_diameter = filtered_df['diameter (km)'].sum()
print(f"Final Answer: {total_diameter}")