import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'elevation (m)' to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
# Calculate the difference between max and min elevation
elevation_diff = df['elevation (m)'].max() - df['elevation (m)'].min()
print(f"Final Answer: {elevation_diff}")