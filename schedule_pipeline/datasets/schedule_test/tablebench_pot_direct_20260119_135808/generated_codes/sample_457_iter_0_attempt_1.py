import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to numeric
df['area km square'] = pd.to_numeric(df['area km square'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['population density per km square'] = pd.to_numeric(df['population density per km square'], errors='coerce')
df['hdi (2011)'] = pd.to_numeric(df['hdi (2011)'], errors='coerce')

# Check for outliers in population density (high values)
density_outliers = df[df['population density per km square'] > 10000]

# Check for missing HDI data
missing_hdi = df[df['hdi (2011)'].isna()]

# Combine findings: Macau has both extremely high density and missing HDI
unusual_country = None
if not density_outliers.empty:
    unusual_country = density_outliers.iloc[0]['country']
elif not missing_hdi.empty:
    unusual_country = missing_hdi.iloc[0]['country']

print(f"Final Answer: {unusual_country}")