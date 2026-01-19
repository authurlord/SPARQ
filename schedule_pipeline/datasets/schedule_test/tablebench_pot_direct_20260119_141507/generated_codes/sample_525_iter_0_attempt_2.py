import pandas as pd

df = pd.read_csv('table.csv')
# Calculate population density: pop / land
df['density'] = df['pop (2010)'].astype(float) / df['land ( sqmi )'].astype(float)

# Find the township with the highest density
max_density_township = df.loc[df['density'].idxmax(), 'township']
print(f"Final Answer: {max_density_township}")