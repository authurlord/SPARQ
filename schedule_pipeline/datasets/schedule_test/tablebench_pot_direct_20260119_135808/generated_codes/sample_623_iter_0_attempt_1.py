import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' and 'Depth (km)' to numeric
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
df['Depth (km)'] = pd.to_numeric(df['Depth (km)'])

# Find the max and min magnitude rows
max_magnitude_row = df.loc[df['Magnitude'].idxmax()]
min_magnitude_row = df.loc[df['Magnitude'].idxmin()]

# Extract depths
max_depth = max_magnitude_row['Depth (km)']
min_depth = min_magnitude_row['Depth (km)']

# Calculate difference
difference = max_depth - min_depth
print(f"Final Answer: {difference}")