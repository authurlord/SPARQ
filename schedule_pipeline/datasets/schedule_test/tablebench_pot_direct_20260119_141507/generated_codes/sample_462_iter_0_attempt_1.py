import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric (handling potential non-numeric strings)
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')
df['exports (usd mn) 2011'] = pd.to_numeric(df['exports (usd mn) 2011'], errors='coerce')
df['2010 population (000)'] = pd.to_numeric(df['2010 population (000)'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Compute mean and std for key variables
gdp_per_capita_mean = df['2008 gdp per capita (usd) a'].mean()
gdp_per_capita_std = df['2008 gdp per capita (usd) a'].std()
exports_mean = df['exports (usd mn) 2011'].mean()
exports_std = df['exports (usd mn) 2011'].std()
pop_mean = df['2010 population (000)'].mean()
pop_std = df['2010 population (000)'].std()

# Identify outliers using 2-standard-deviation rule
outliers_gdp = df[(df['2008 gdp per capita (usd) a'] < (gdp_per_capita_mean - 2 * gdp_per_capita_std)) |
                  (df['2008 gdp per capita (usd) a'] > (gdp_per_capita_mean + 2 * gdp_per_capita_std))]

outliers_exports = df[(df['exports (usd mn) 2011'] < (exports_mean - 2 * exports_std)) |
                      (df['exports (usd mn) 2011'] > (exports_mean + 2 * exports_std))]

outliers_pop = df[(df['2010 population (000)'] < (pop_mean - 2 * pop_std)) |
                  (df['2010 population (000)'] > (pop_mean + 2 * pop_std))]

# Combine outlier provinces (by district name)
outlier_provinces = set()
for idx, row in outliers_gdp.iterrows():
    outlier_provinces.add(row['district'])
for idx, row in outliers_exports.iterrows():
    outlier_provinces.add(row['district'])
for idx, row in outliers_pop.iterrows():
    outlier_provinces.add(row['district'])

# Final answer: list of provinces with abnormal data points
final_outliers = list(outlier_provinces)
print(f"Final Answer: {', '.join(final_outliers)}")