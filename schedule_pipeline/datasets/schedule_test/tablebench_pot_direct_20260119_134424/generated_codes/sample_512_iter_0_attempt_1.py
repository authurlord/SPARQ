import pandas as pd

df = pd.read_csv('table.csv')

# Drop rows with missing values in 'Length (miles)' or 'Locks'
df.dropna(subset=['Length (miles)', 'Locks'], inplace=True)

# Convert columns to numeric
df['Length (miles)'] = pd.to_numeric(df['Length (miles)'])
df['Locks'] = pd.to_numeric(df['Locks'])

# Calculate lock density
df['Lock Density'] = df['Locks'] / df['Length (miles)']

# Find the canal with the highest lock density
max_density_canal = df.loc[df['Lock Density'].idxmax()]['Canal']

print(f"Final Answer: {max_density_canal}")