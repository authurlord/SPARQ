import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select rows where both 21st and 22nd ionization energies are not null
valid_rows = df.dropna(subset=['21st', '22nd'])

# Convert 21st and 22nd to numeric (handle any potential string issues)
valid_rows['21st'] = pd.to_numeric(valid_rows['21st'], errors='coerce')
valid_rows['22nd'] = pd.to_numeric(valid_rows['22nd'], errors='coerce')

# Filter out rows where either value is still NaN
valid_rows = valid_rows.dropna(subset=['21st', '22nd'])

# Compute the ratio: 22nd / 21st
valid_rows['ratio'] = valid_rows['22nd'] / valid_rows['21st']

# Find the element with the highest ratio
max_ratio_row = valid_rows.loc[valid_rows['ratio'].idxmax()]
element_name = max_ratio_row['name']

print(f"Final Answer: {element_name}")