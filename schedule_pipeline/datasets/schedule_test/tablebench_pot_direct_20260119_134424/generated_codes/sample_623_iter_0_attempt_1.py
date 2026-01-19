import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' to float for comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Find the max and min magnitude rows
max_depth = df.loc[df['Magnitude'].idxmax()]['Depth (km)']
min_depth = df.loc[df['Magnitude'].idxmin()]['Depth (km)']
# Convert depth to float
max_depth = float(max_depth)
min_depth = float(min_depth)
# Calculate difference
difference = max_depth - min_depth
print(f"Final Answer: {difference}")