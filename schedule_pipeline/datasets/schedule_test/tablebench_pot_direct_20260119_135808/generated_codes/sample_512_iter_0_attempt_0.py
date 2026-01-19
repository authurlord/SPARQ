import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Length (miles)' and 'Locks' to numeric, handling missing values
df['Length (miles)'] = pd.to_numeric(df['Length (miles)'], errors='coerce')
df['Locks'] = pd.to_numeric(df['Locks'], errors='coerce')

# Calculate lock density
df['Lock Density'] = df['Locks'] / df['Length (miles)']

# Find the canal with the highest lock density
max_density_canal = df.loc[df['Lock Density'].idxmax(), 'Canal']

print(f"Final Answer: {max_density_canal}")