import pandas as pd

df = pd.read_csv('table.csv')

# Convert '21st' and '22nd' columns to numeric, coercing errors to NaN
df['21st'] = pd.to_numeric(df['21st'], errors='coerce')
df['22nd'] = pd.to_numeric(df['22nd'], errors='coerce')

# Filter rows where both 21st and 22nd ionization energies are available
valid_data = df.dropna(subset=['21st', '22nd'])

# Compute the ratio of 22nd to 21st ionization energy
valid_data['ratio'] = valid_data['22nd'] / valid_data['21st']

# Find the element with the highest ratio
max_ratio_element = valid_data.loc[valid_data['ratio'].idxmax()]

print(f"Final Answer: {max_ratio_element['name']}")