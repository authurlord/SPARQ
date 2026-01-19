import pandas as pd

df = pd.read_csv('table.csv')
# Find the place with the largest and smallest area
max_area_row = df.loc[df['area (km 2 )'].idxmax()]
min_area_row = df.loc[df['area (km 2 )'].idxmin()]

largest_area = float(max_area_row['area (km 2 )'])
smallest_area = float(min_area_row['area (km 2 )'])
area_difference = largest_area - smallest_area

print(f"Final Answer: {max_area_row['place']}, {area_difference:.2f}")