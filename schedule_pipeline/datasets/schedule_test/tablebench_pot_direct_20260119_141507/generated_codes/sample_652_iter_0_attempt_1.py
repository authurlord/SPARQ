import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract the relevant columns
pop_density = df['pop density ( / km square)']
gdp_per_capita = df['gdp per capita usd (2009 - 2011)']

# Calculate correlation coefficient
correlation = pop_density.corr(gdp_per_capita)

# Detect outliers using IQR method
Q1 = pop_density.quantile(0.25)
Q3 = pop_density.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(pop_density < lower_bound) | (pop_density > upper_bound)]
outliers_list = outliers.index.tolist()

print(f"Final Answer: {correlation:.3f}, {outliers_list}")