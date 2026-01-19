import pandas as pd

df = pd.read_csv('table.csv')
# Extract elevation values
elevations = df['elevation (m)'].astype(int)
max_elevation = elevations.max()
min_elevation = elevations.min()
difference = max_elevation - min_elevation
print(f"Final Answer: {difference}")