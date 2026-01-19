import pandas as pd

df = pd.read_csv('table.csv')
# Calculate population density
df['population_density'] = df['pop (2010)'] / df['land ( sqmi )']
# Compute correlation between latitude and population density
correlation = df['latitude'].corr(df['population_density'])
print(f"Final Answer: {correlation:.3f}")