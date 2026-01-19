import pandas as pd

df = pd.read_csv('table.csv')
# Extract elevation values and compute the difference between max and min
max_elevation = df['elevation (m)'].max()
min_elevation = df['elevation (m)'].min()
elevation_difference = max_elevation - min_elevation
print(f"Final Answer: {elevation_difference}")