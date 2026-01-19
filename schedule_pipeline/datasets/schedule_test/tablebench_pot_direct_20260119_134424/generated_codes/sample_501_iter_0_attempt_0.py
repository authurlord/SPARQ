import pandas as pd

df = pd.read_csv('table.csv')
# Drop rows where either 21st or 22nd ionization energy is missing
df_filtered = df.dropna(subset=['21st', '22nd'])
# Convert '21st' and '22nd' columns to numeric
df_filtered['21st'] = pd.to_numeric(df_filtered['21st'], errors='coerce')
df_filtered['22nd'] = pd.to_numeric(df_filtered['22nd'], errors='coerce')
# Calculate the ratio of 22nd to 21st ionization energy
df_filtered['ratio'] = df_filtered['22nd'] / df_filtered['21st']
# Find the element with the highest ratio
max_ratio_element = df_filtered.loc[df_filtered['ratio'].idxmax()]
print(f"Final Answer: {max_ratio_element['name']}")