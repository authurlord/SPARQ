import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop rows where 'Finalists' is None (missing)
df_clean = df.dropna(subset=['Finalists'])

# Convert relevant columns to numeric (in case of strings like '9.267')
df_clean[['Swimsuit', 'Interview', 'Evening Gown', 'Finalists']] = df_clean[['Swimsuit', 'Interview', 'Evening Gown', 'Finalists']].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN after conversion
df_clean = df_clean.dropna()

# Compute correlation with 'Finalists'
correlations = df_clean[['Swimsuit', 'Interview', 'Evening Gown', 'Finalists']].corr()['Finalists'].abs()

# Check which factors have correlation > 0.7 (significant influence)
significant_factors = []
for col in ['Swimsuit', 'Interview', 'Evening Gown']:
    if abs(correlations[col]) > 0.7:
        significant_factors.append(col)

if significant_factors:
    Final Answer: , '.join(significant_factors)
else:
    Final Answer: no clear impact