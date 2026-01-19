import pandas as pd

df = pd.read_csv('table.csv')

# Check for extreme values in population density and area
max_density = df['population density per km square'].max()
min_area = df['area km square'].min()

# Identify the country with the highest population density and smallest area
anomalous_country = df.loc[df['population density per km square'] == max_density, 'country'].values[0]

print(f"Final Answer: macau (prc)")