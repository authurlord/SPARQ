import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop rows where length is missing (NaN or None)
df_filtered = df.dropna(subset=['Length (miles)'])

# Convert 'Length (miles)' and 'Locks' to numeric (in case of string values)
df_filtered['Length (miles)'] = pd.to_numeric(df_filtered['Length (miles)'], errors='coerce')
df_filtered['Locks'] = pd.to_numeric(df_filtered['Locks'], errors='coerce')

# Calculate lock density: locks / length
df_filtered['lock_density'] = df_filtered['Locks'] / df_filtered['Length (miles)']

# Find the canal with the highest lock density
max_density_canal = df_filtered.loc[df_filtered['lock_density'].idxmax(), 'Canal']
print(f"Final Answer: {max_density_canal}")