import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Magnitude' to float for comparison
df['Magnitude'] = df['Magnitude'].str.replace('7.', '7.0').astype(float)

# Find the max and min magnitude
max_magnitude = df['Magnitude'].max()
min_magnitude = df['Magnitude'].min()

# Get the corresponding depth values
max_depth = df[df['Magnitude'] == max_magnitude]['Depth (km)'].iloc[0]
min_depth = df[df['Magnitude'] == min_magnitude]['Depth (km)'].iloc[0]

# Calculate the difference
depth_difference = max_depth - min_depth

print(f"Final Answer: {depth_difference}")