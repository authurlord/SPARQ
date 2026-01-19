import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'pop (2010)' and 'land ( sqmi )' to numeric, handling any potential issues
df['pop (2010)'] = pd.to_numeric(df['pop (2010)'], errors='coerce')
df['land ( sqmi )'] = pd.to_numeric(df['land ( sqmi )'], errors='coerce')

# Compute population density (population per square mile)
df['density'] = df['pop (2010)'] / df['land ( sqmi )']

# Extract latitude and density for correlation analysis
latitude = df['latitude'].astype(float)
density = df['density']

# Compute correlation between latitude and density
correlation = latitude.corr(density)

print(f"Final Answer: {correlation:.3f}")