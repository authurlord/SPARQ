import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for calculation
df['pop (2010)'] = pd.to_numeric(df['pop (2010)'])
df['land ( sqmi )'] = pd.to_numeric(df['land ( sqmi )'])

# Calculate population density
df['population_density'] = df['pop (2010)'] / df['land ( sqmi )']

# Find the township with the highest population density
max_density_township = df.loc[df['population_density'].idxmax(), 'township']

print(f"Final Answer: {max_density_township}")