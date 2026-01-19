import pandas as pd

df = pd.read_csv('table.csv')
# Find max and min external area
max_area_row = df.loc[df['a external (cm 2 )'].idxmax()]
min_area_row = df.loc[df['a external (cm 2 )'].idxmin()]

# Extract maximum pressure values
max_pressure = max_area_row['p max ( bar )']
min_pressure = min_area_row['p max ( bar )']

# Calculate the difference
pressure_difference = max_pressure - min_pressure
print(f"Final Answer: {pressure_difference}")