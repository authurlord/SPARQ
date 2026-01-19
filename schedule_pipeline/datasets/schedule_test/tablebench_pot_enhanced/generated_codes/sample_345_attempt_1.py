import pandas as pd

df = pd.read_csv('table.csv')
# Convert population density to numeric
df['pop density (per km square)'] = pd.to_numeric(df['pop density (per km square)'])
# Filter cities with population density > 3000
high_density_cities = df[df['pop density (per km square)'] > 3000]
# Count the number of such cities
count = len(high_density_cities)
print(f"Final Answer: {count}")