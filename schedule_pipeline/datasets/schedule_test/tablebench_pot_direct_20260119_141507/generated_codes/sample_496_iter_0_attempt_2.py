import pandas as pd

df = pd.read_csv('table.csv')
# Convert height and floors to numeric
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['floors'] = pd.to_numeric(df['floors'], errors='coerce')

# Calculate floor-to-height ratio
df['floor_to_height_ratio'] = df['floors'] / df['height']

# Find the building with the highest ratio
max_ratio_building = df.loc[df['floor_to_height_ratio'].idxmax(), 'building']

print(f"Final Answer: {max_ratio_building}")