import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'area (km 2 )' to float for numerical operations
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'])
# Find the place with the largest area
max_area_place = df.loc[df['area (km 2 )'].idxmax(), 'place']
# Find the place with the smallest area
min_area_place = df.loc[df['area (km 2 )'].idxmin(), 'place']
# Calculate the difference
difference = df['area (km 2 )'].max() - df['area (km 2 )'].min()
print(f"Final Answer: {max_area_place}, {difference:.2f}")