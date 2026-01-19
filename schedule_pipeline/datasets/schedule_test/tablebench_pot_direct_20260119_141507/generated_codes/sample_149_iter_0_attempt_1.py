import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'pop (2010)' and 'land ( sqmi )' to numeric
df['pop (2010)'] = pd.to_numeric(df['pop (2010)'], errors='coerce')
df['land ( sqmi )'] = pd.to_numeric(df['land ( sqmi )'], errors='coerce')

# Calculate population density (per sqmi)
df['density'] = df['pop (2010)'] / df['land ( sqmi )']

# Drop rows with NaN due to invalid data
df = df.dropna(subset=['density'])

# Compute correlation between latitude and density
latitude = df['latitude'].astype(float)
density = df['density']

correlation = latitude.corr(density)

print(f"Final Answer: {correlation:.3f}")