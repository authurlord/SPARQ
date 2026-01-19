import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where both 21st and 22nd ionization energies are available
df_filtered = df.dropna(subset=['21st', '22nd'])

# Convert '21st' and '22nd' columns to numeric
df_filtered['21st'] = pd.to_numeric(df_filtered['21st'])
df_filtered['22nd'] = pd.to_numeric(df_filtered['22nd'])

# Calculate the ionization energy ratio from 21st to 22nd level
df_filtered['ratio'] = df_filtered['22nd'] / df_filtered['21st']

# Find the element with the highest ratio
max_ratio_row = df_filtered.loc[df_filtered['ratio'].idxmax()]

# Output the name of the element
print(f"Final Answer: {max_ratio_row['name']}")