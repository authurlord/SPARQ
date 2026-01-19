import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Length (miles)' and 'Locks' to numeric, handling missing values
df['Length (miles)'] = pd.to_numeric(df['Length (miles)'], errors='coerce')
df['Locks'] = pd.to_numeric(df['Locks'], errors='coerce')

# Calculate lock density (locks / length)
df['Lock Density'] = df['Locks'] / df['Length (miles)']

# Find the canal with the highest lock density
max_density_index = df['Lock Density'].idxmax()
canal_with_highest_density = df.loc[max_density_index, 'Canal']

print(f"Final Answer: {canal_with_highest_density}")