import pandas as pd

df = pd.read_csv('table.csv')
# Find the maximum and minimum area values
max_area = df['area (km 2 )'].max()
min_area = df['area (km 2 )'].min()
difference = max_area - min_area

# Get the place with the largest area
largest_place = df.loc[df['area (km 2 )'].idxmax(), 'place']

print(f"Final Answer: {largest_place}, {difference:.2f}")