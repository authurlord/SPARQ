import pandas as pd

df = pd.read_csv('table.csv')
# Find the max and min external areas
max_area_row = df.loc[df['a external (cm 2 )'].idxmax()]
min_area_row = df.loc[df['a external (cm 2 )'].idxmin()]
# Get their p max values
max_pressure = float(max_area_row['p max ( bar )'])
min_pressure = float(min_area_row['p max ( bar )'])
# Calculate difference
pressure_difference = max_pressure - min_pressure
print(f"Final Answer: {pressure_difference}")