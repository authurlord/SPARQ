import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where both 21st and 22nd are not null
valid_rows = df.dropna(subset=['21st', '22nd'])

# Compute the ionization energy ratio (22nd / 21st)
valid_rows['ratio'] = valid_rows['22nd'] / valid_rows['21st']

# Find the element with the highest ratio
max_ratio_row = valid_rows.loc[valid_rows['ratio'].idxmax()]
element_name = max_ratio_row['name']

print(f"Final Answer: {element_name}")