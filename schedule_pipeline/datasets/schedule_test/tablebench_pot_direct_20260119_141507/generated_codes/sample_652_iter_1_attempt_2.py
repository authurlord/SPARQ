import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert the required columns to numeric, coercing errors to NaN
df['pop density ( / km square)'] = pd.to_numeric(df['pop density ( / km square)'], errors='coerce')
df['gdp per capita usd (2009 - 2011)'] = pd.to_numeric(df['gdp per capita usd (2009 - 2011)'], errors='coerce')

# Drop rows with missing values in either column
df_clean = df.dropna(subset=['pop density ( / km square)', 'gdp per capita usd (2009 - 2011)'])

# Calculate correlation coefficient
correlation = df_clean['pop density ( / km square)'].corr(df_clean['gdp per capita usd (2009 - 2011)'])

# Identify outliers using IQR method
Q1 = df_clean['pop density ( / km square)'].quantile(0.25)
Q3 = df_clean['pop density ( / km square)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[
    (df_clean['pop density ( / km square)'] < lower_bound) |
    (df_clean['pop density ( / km square)'] > upper_bound)
]

# Extract country names of outliers
outlier_countries = outliers['country / territory'].tolist()

print(f"Final Answer: {correlation:.3f}, {', '.join(outlier_countries) if outlier_countries else 'none'}")