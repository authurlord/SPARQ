import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area (km square)'] = pd.to_numeric(df['area (km square)'])
df['pop'] = pd.to_numeric(df['pop'])

# Calculate population density
df['pop_density'] = df['pop'] / df['area (km square)']

# Group by subregion and calculate average population density
avg_density_by_subregion = df.groupby('subregion')['pop_density'].mean()

# Find the subregion with the highest average population density
highest_density_subregion = avg_density_by_subregion.idxmax()

print(f"Final Answer: {highest_density_subregion}")