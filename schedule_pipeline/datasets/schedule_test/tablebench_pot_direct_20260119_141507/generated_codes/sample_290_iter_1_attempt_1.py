import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'a external (cm 2 )' to float, handling any formatting issues
df['a external (cm 2 )'] = pd.to_numeric(df['a external (cm 2 )'], errors='coerce')

# Find the row with maximum and minimum external area
max_area_row = df.loc[df['a external (cm 2 )'].idxmax()]
min_area_row = df.loc[df['a external (cm 2 )'].idxmin()]

# Extract the maximum pressure values
max_pressure = max_area_row['p max ( bar )']
min_pressure = min_area_row['p max ( bar )']

# Compute the difference
pressure_difference = max_pressure - min_pressure

print(f"Final Answer: {pressure_difference}")