import pandas as pd

df = pd.read_csv('table.csv')

# Calculate population density (pop / land area)
df['population_density'] = df['pop (2010)'].astype(float) / df['land ( sqmi )'].astype(float)

# Calculate the correlation between latitude and population density
correlation = df['latitude'].astype(float).corr(df['population_density'])

print(f"Final Answer: {correlation:.3f}")