import pandas as pd

df = pd.read_csv('table.csv')
# Filter craters with diameter > 40 km and sum their diameters
total_diameter = df[df['diameter (km)'] > 40]['diameter (km)'].sum()
print(f"Final Answer: {total_diameter}")