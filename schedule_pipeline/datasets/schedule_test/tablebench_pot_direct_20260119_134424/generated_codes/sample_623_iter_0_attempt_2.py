import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' to float for proper comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Find the depth of the earthquake with the highest magnitude
max_depth = df.loc[df['Magnitude'].idxmax(), 'Depth (km)']
# Find the depth of the earthquake with the lowest magnitude
min_depth = df.loc[df['Magnitude'].idxmin(), 'Depth (km)']
# Convert depths to float and calculate the difference
depth_diff = abs(float(max_depth) - float(min_depth))
print(f"Final Answer: {depth_diff:.1f}")