import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'elevation (m)' to integer
df['elevation (m)'] = df['elevation (m)'].astype(int)
# Calculate the difference between highest and lowest elevation
elevation_diff = df['elevation (m)'].max() - df['elevation (m)'].min()
print(f"Final Answer: {elevation_diff}")