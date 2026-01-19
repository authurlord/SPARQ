import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' and 'Depth (km)' to float for numerical operations
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
df['Depth (km)'] = pd.to_numeric(df['Depth (km)'])

# Find the max and min magnitude values
max_magnitude_row = df.loc[df['Magnitude'].idxmax()]
min_magnitude_row = df.loc[df['Magnitude'].idxmin()]

# Calculate the difference in depth
depth_difference = abs(max_magnitude_row['Depth (km)'] - min_magnitude_row['Depth (km)'])

print(f"Final Answer: {depth_difference}")