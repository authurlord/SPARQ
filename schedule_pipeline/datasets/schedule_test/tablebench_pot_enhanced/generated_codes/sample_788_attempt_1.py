import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Greece (including regions within Greece)
greece_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to numeric
greece_mountains['elevation (m)'] = pd.to_numeric(greece_mountains['elevation (m)'])
# Calculate median elevation
median_elevation = greece_mountains['elevation (m)'].median()
print(f"Final Answer: {median_elevation}")