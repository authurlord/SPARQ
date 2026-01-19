import pandas as pd

df = pd.read_csv('table.csv')

# Find the depth for the earthquake with highest magnitude (7.6)
max_magnitude_row = df[df['Magnitude'] == '7.6']
depth_max = float(max_magnitude_row['Depth (km)'].iloc[0])

# Find the depth for the earthquake with lowest magnitude (7.0)
min_magnitude_row = df[df['Magnitude'] == '7.0']
depth_min = float(min_magnitude_row['Depth (km)'].iloc[0])

# Calculate the difference
depth_difference = depth_max - depth_min
print(f"Final Answer: {depth_difference:.1f}")