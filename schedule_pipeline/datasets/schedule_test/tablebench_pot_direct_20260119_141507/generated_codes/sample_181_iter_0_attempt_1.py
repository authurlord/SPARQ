import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop rows where 'Finalists' is None (missing)
df_clean = df.dropna(subset=['Finalists'])

# Convert 'Finalists' and other score columns to numeric (in case of string formatting issues)
df_clean['Finalists'] = pd.to_numeric(df_clean['Finalists'], errors='coerce')
df_clean['Swimsuit'] = pd.to_numeric(df_clean['Swimsuit'], errors='coerce')
df_clean['Interview'] = pd.to_numeric(df_clean['Interview'], errors='coerce')
df_clean['Evening Gown'] = pd.to_numeric(df_clean['Evening Gown'], errors='coerce')

# Compute correlation with 'Finalists'
correlations = {
    'Swimsuit': df_clean['Finalists'].corr(df_clean['Swimsuit']),
    'Interview': df_clean['Finalists'].corr(df_clean['Interview']),
    'Evening Gown': df_clean['Finalists'].corr(df_clean['Evening Gown'])
}

# Check if any correlation has absolute value > 0.7 (considered significant)
significant_factors = []
for factor, corr in correlations.items():
    if abs(corr) > 0.7:
        significant_factors.append(factor)

if significant_factors:
    Final Answer: , '.join(significant_factors)
else:
    Final Answer: no clear impact