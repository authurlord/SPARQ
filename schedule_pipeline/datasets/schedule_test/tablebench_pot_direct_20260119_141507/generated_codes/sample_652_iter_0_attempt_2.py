import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract the columns of interest
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

outliers_pop_density = pop_density[(pop_density < lower_bound) | (pop_density > upper_bound)]

# Check if any outlier exists in gdp_per_capita with corresponding pop_density
outlier_indices = []
for idx, row in df.iterrows():
    if (row['pop density ( / km square)'] < lower_bound) or (row['pop density ( / km square)'] > upper_bound):
        outlier_indices.append(idx)

# If there are outliers, print them; otherwise, indicate none
outliers_present = len(outlier_indices) > 0
outlier_info = f"Outliers present: {outliers_present}, Correlation coefficient: {correlation:.3f}"

print(f"Final Answer: {correlation:.3f}, {outliers_present}")