import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'a external (cm 2 )' to float for comparison
df['a external (cm 2 )'] = pd.to_numeric(df['a external (cm 2 )'], errors='coerce')
# Find the max and min external areas
max_area_row = df.loc[df['a external (cm 2 )'].idxmax()]
min_area_row = df.loc[df['a external (cm 2 )'].idxmin()]
# Get the corresponding p max values
max_pressure = max_area_row['p max ( bar )']
min_pressure = min_area_row['p max ( bar )']
# Calculate the difference
pressure_difference = abs(float(max_pressure) - float(min_pressure))
print(f"Final Answer: {pressure_difference}")