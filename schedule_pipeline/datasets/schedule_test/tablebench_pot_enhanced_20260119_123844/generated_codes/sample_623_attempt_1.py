import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' to float for proper comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Find the max and min magnitude values
max_magnitude_depth = df.loc[df['Magnitude'].idxmax(), 'Depth (km)']
min_magnitude_depth = df.loc[df['Magnitude'].idxmin(), 'Depth (km)']
# Convert depths to float and calculate difference
depth_difference = abs(float(max_magnitude_depth) - float(min_magnitude_depth))
print(f"Final Answer: {depth_difference:.1f}")