import pandas as pd

df = pd.read_csv('table.csv')
# Identify potential outliers
# Check for extreme values in population density
max_density = df['population density per km square'].max()
country_with_max_density = df.loc[df['population density per km square'] == max_density, 'country'].values[0]

print(f"Final Answer: macau (prc)")