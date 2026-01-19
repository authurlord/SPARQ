import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'elevation (m)' to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
# Find max and min elevation
max_elevation = df['elevation (m)'].max()
min_elevation = df['elevation (m)'].min()
# Calculate difference
difference = max_elevation - min_elevation
print(f"Final Answer: {difference}")