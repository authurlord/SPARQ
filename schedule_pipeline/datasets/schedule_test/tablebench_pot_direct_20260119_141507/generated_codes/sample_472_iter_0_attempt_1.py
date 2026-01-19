import pandas as pd

df = pd.read_csv('table.csv')
# Find the district with the highest population density
max_density = df['pop density (per km2)'].max()
district_with_max_density = df[df['pop density (per km2)'] == max_density]['district'].values[0]
print(f"Final Answer: {district_with_max_density}")