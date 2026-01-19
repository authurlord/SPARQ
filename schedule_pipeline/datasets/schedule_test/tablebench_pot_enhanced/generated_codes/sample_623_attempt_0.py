import pandas as pd

df = pd.read_csv('table.csv')
# Convert Magnitude and Depth to float for numerical operations
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
df['Depth (km)'] = pd.to_numeric(df['Depth (km)'])

# Find the max and min magnitude values
max_magnitude_row = df.loc[df['Magnitude'].idxmax()]
min_magnitude_row = df.loc[df['Magnitude'].idxmin()]

# Get the depth values
max_depth = max_magnitude_row['Depth (km)']
min_depth = min_magnitude_row['Depth (km)']

# Calculate the difference
difference = max_depth - min_depth
print(f"Final Answer: {difference}")