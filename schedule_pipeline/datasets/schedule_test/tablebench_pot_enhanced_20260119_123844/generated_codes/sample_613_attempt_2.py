import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'elevation (m)' column to integers
elevations = df['elevation (m)'].astype(int)
# Calculate the difference between highest and lowest elevation
elevation_difference = elevations.max() - elevations.min()
print(f"Final Answer: {elevation_difference}")