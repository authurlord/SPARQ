import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')
df['exports (usd mn) 2011'] = pd.to_numeric(df['exports (usd mn) 2011'], errors='coerce')

# Drop rows with missing values
df_clean = df.dropna(subset=['2008 gdp per capita (usd) a', 'exports (usd mn) 2011'])

# Calculate IQR for GDP per capita and exports
Q1_gdp = df_clean['2008 gdp per capita (usd) a'].quantile(0.25)
Q3_gdp = df_clean['2008 gdp per capita (usd) a'].quantile(0.75)
IQR_gdp = Q3_gdp - Q1_gdp

Q1_export = df_clean['exports (usd mn) 2011'].quantile(0.25)
Q3_export = df_clean['exports (usd mn) 2011'].quantile(0.75)
IQR_export = Q3_export - Q1_export

# Identify outliers
outliers_gdp = df_clean[
    (df_clean['2008 gdp per capita (usd) a'] < (Q1_gdp - 1.5 * IQR_gdp)) |
    (df_clean['2008 gdp per capita (usd) a'] > (Q3_gdp + 1.5 * IQR_gdp))
]

outliers_export = df_clean[
    (df_clean['exports (usd mn) 2011'] < (Q1_export - 1.5 * IQR_export)) |
    (df_clean['exports (usd mn) 2011'] > (Q3_export + 1.5 * IQR_export))
]

# Combine outliers and get district names
abnormal_provinces = set()
for _, row in outliers_gdp.iterrows():
    abnormal_provinces.add(row['district'])
for _, row in outliers_export.iterrows():
    abnormal_provinces.add(row['district'])

# Final list of abnormal provinces
abnormal_list = list(abnormal_provinces)
print(f"Final Answer: {', '.join(abnormal_list)}")