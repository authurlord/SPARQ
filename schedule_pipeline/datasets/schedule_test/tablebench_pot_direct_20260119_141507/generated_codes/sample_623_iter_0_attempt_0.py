import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' to float for proper comparison
df['Magnitude'] = df['Magnitude'].str.strip().astype(float)

# Find the maximum and minimum magnitude
max_magnitude = df['Magnitude'].max()
min_magnitude = df['Magnitude'].min()

# Get the depth values for the earthquakes with max and min magnitude
max_depth = df[df['Magnitude'] == max_magnitude]['Depth (km)'].values[0]
min_depth = df[df['Magnitude'] == min_magnitude]['Depth (km)'].values[0]

# Calculate the difference in depth
depth_difference = abs(max_depth - min_depth)
print(f"Final Answer: {depth_difference}")