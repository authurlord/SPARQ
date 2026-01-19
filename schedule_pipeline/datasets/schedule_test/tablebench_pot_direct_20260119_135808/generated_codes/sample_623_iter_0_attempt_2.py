import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' to float for comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Find the depth of the earthquake with the highest magnitude
max_magnitude_depth = df.loc[df['Magnitude'].idxmax(), 'Depth (km)']
# Find the depth of the earthquake with the lowest magnitude
min_magnitude_depth = df.loc[df['Magnitude'].idxmin(), 'Depth (km)']
# Convert depths to float and calculate difference
depth_difference = abs(float(max_magnitude_depth) - float(min_magnitude_depth))
print(f"Final Answer: {depth_difference:.1f}")