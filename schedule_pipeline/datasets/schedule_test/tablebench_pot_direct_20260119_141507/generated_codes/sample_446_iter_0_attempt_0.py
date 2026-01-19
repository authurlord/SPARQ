import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert population and GDP per capita to numeric
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Drop rows with NaN due to parsing errors
df = df.dropna(subset=['population', 'gdp per capita (us)'])

# Calculate mean and std for population and GDP per capita
pop_mean = df['population'].mean()
pop_std = df['population'].std()
gdp_per_capita_mean = df['gdp per capita (us)'].mean()
gdp_per_capita_std = df['gdp per capita (us)'].std()

# Define thresholds for outliers (3 standard deviations)
pop_lower = pop_mean - 3 * pop_std
pop_upper = pop_mean + 3 * pop_std
gdp_per_capita_lower = gdp_per_capita_mean - 3 * gdp_per_capita_std
gdp_per_capita_upper = gdp_per_capita_mean + 3 * gdp_per_capita_std

# Identify outliers in population or GDP per capita
outliers = []
for idx, row in df.iterrows():
    country = row['member countries']
    pop = row['population']
    gdp_per_capita = row['gdp per capita (us)']
    
    if (pop < pop_lower or pop > pop_upper) or (gdp_per_capita < gdp_per_capita_lower or gdp_per_capita > gdp_per_capita_upper):
        outliers.append(country)

# Remove duplicates and sort
unique_outliers = sorted(list(set(outliers)))
print(f"Final Answer: {', '.join(unique_outliers)}")