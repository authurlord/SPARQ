import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' and 'Depth (km)' to float for numerical operations
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
df['Depth (km)'] = pd.to_numeric(df['Depth (km)'])

# Find the max and min magnitude values
max_magnitude_depth = df.loc[df['Magnitude'].idxmax()]['Depth (km)']
min_magnitude_depth = df.loc[df['Magnitude'].idxmin()]['Depth (km)']

# Calculate the difference
depth_difference = max_magnitude_depth - min_magnitude_depth
print(f"Final Answer: {depth_difference:.1f}")