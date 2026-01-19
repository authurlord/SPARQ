import pandas as pd

df = pd.read_csv('table.csv')
# Convert height and floors to float for calculation
df['height'] = pd.to_numeric(df['height'])
df['floors'] = pd.to_numeric(df['floors'])
# Calculate Floor-to-Height Ratio
df['floor_to_height_ratio'] = df['floors'] / df['height']
# Find the building with the highest ratio
max_ratio_building = df.loc[df['floor_to_height_ratio'].idxmax(), 'building']
print(f"Final Answer: {max_ratio_building}")