import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert population and GDP per capita to numeric
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Remove rows with NaN due to conversion errors
df = df.dropna(subset=['population', 'gdp per capita (us)'])

# Calculate mean and std for population and gdp per capita
mean_pop = df['population'].mean()
std_pop = df['population'].std()
mean_gdp = df['gdp per capita (us)'].mean()
std_gdp = df['gdp per capita (us)'].std()

# Identify outliers using z-score (|z| > 2)
outliers_pop = df[np.abs((df['population'] - mean_pop) / std_pop) > 2]
outliers_gdp = df[np.abs((df['gdp per capita (us)'] - mean_gdp) / std_gdp) > 2]

# Combine unique country names from both outlier sets
outlier_countries = set()
for idx, row in outliers_pop.iterrows():
    outlier_countries.add(row['member countries'])
for idx, row in outliers_gdp.iterrows():
    outlier_countries.add(row['member countries'])

# Final answer: list of countries with significant deviation
final_answer = list(outlier_countries)
print(f"Final Answer: {', '.join(final_answer)}")