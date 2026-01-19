import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter rows where both 21st and 22nd have non-null values
valid_data = df.dropna(subset=['21st', '22nd'])

# Convert 21st and 22nd to numeric (some values are strings with decimal)
valid_data['21st'] = pd.to_numeric(valid_data['21st'], errors='coerce')
valid_data['22nd'] = pd.to_numeric(valid_data['22nd'], errors='coerce')

# Remove rows where either value is still NaN
valid_data = valid_data.dropna(subset=['21st', '22nd'])

# Calculate the Ionization Energy Ratio (22nd / 21st)
valid_data['ratio'] = valid_data['22nd'] / valid_data['21st']

# Find the element with the highest ratio
max_ratio_row = valid_data.loc[valid_data['ratio'].idxmax()]
element_with_max_ratio = max_ratio_row['name']

print(f"Final Answer: {element_with_max_ratio}")