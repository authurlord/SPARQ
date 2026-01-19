import pandas as pd

df = pd.read_csv('table.csv')

# Correctly access the columns using proper naming
# The column names have spaces and special characters; we use exact names
external_area_col = 'a external (cm 2 )'
pressure_col = 'p max ( bar )'

# Convert external area to float for comparison
df['a external (cm 2 )'] = df['a external (cm 2 )'].str.replace(' ', '').astype(float)

# Find the row with maximum and minimum external area
max_area_row = df.loc[df['a external (cm 2 )'].idxmax()]
min_area_row = df.loc[df['a external (cm 2 )'].idxmin()]

max_pressure = max_area_row['p max ( bar )']
min_pressure = min_area_row['p max ( bar )']

difference = max_pressure - min_pressure
print(f"Final Answer: {difference}")