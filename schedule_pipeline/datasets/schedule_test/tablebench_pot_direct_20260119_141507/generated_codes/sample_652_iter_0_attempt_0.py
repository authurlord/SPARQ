import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract the columns of interest
pop_density = df['pop density ( / km square)']
gdp_per_capita = df['gdp per capita usd (2009 - 2011)']

# Calculate correlation coefficient
correlation = pop_density.corr(gdp_per_capita)

# Identify outliers using IQR method
Q1 = pop_density.quantile(0.25)
Q3 = pop_density.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = []
for i, (density, gdp) in enumerate(zip(pop_density, gdp_per_capita)):
    if density < lower_bound or density > upper_bound:
        outliers.append((i, density, gdp))

# Print results
print(f"Final Answer: {correlation:.3f}, {outliers}")