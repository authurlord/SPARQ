import pandas as pd

df = pd.read_csv('table.csv')

# Correctly access the column with external area (note the exact name)
external_area_col = 'a external (cm 2 )'
pressure_col = 'p max ( bar )'

# Convert the external area column to float for numerical comparison
df['a external (cm 2 )'] = pd.to_numeric(df[external_area_col], errors='coerce')

# Find the max and min external area values and their corresponding p max
max_external_row = df.loc[df[external_area_col].idxmax()]
min_external_row = df.loc[df[external_area_col].idxmin()]

max_pressure = max_external_row[pressure_col]
min_pressure = min_external_row[pressure_col]

difference = max_pressure - min_pressure
print(f"Final Answer: {difference}")