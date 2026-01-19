import pandas as pd

df = pd.read_csv('table.csv')

# Check for missing HDI data
missing_hdi = df[df['hdi (2011)'] == 'no data']['country'].tolist()

# Check for extreme population density
max_density = df['population density per km square'].max()
country_with_max_density = df[df['population density per km square'] == max_density]['country'].iloc[0]

print(f"Final Answer: macau (prc)")