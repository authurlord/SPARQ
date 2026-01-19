import pandas as pd

df = pd.read_csv('table.csv')

# Remove aggregate rows (non-country entries)
df_filtered = df[df['member countries'].str.contains(r'accession|existing', case=False, na=False) == False]

# Convert numeric columns to float for analysis
df_filtered['gdp per capita (us)'] = pd.to_numeric(df_filtered['gdp per capita (us)'], errors='coerce')
df_filtered['gdp (billion us)'] = pd.to_numeric(df_filtered['gdp (billion us)'], errors='coerce')

# Identify outliers using IQR method on GDP per capita
Q1 = df_filtered['gdp per capita (us)'].quantile(0.25)
Q3 = df_filtered['gdp per capita (us)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_gdp_per_capita = df_filtered[(df_filtered['gdp per capita (us)'] < lower_bound) | (df_filtered['gdp per capita (us)'] > upper_bound)]

# Also check for extreme GDP values
outliers_gdp = df_filtered[(df_filtered['gdp (billion us)'] < df_filtered['gdp (billion us)'].quantile(0.01)) | 
                           (df_filtered['gdp (billion us)'] > df_filtered['gdp (billion us)'].quantile(0.99))]

# Combine both outlier conditions
outlier_countries = set(outliers_gdp_per_capita['member countries']) | set(outliers_gdp['member countries'])

# Final answer: list of country names with abnormal data points
print(f"Final Answer: {', '.join(sorted(outlier_countries))}")