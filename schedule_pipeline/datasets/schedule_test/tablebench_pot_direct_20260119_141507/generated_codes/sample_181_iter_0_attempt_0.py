import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert string scores to float and drop rows where 'Finalists' is None
df['Swimsuit'] = pd.to_numeric(df['Swimsuit'], errors='coerce')
df['Interview'] = pd.to_numeric(df['Interview'], errors='coerce')
df['Evening Gown'] = pd.to_numeric(df['Evening Gown'], errors='coerce')
df['Finalists'] = pd.to_numeric(df['Finalists'], errors='coerce')

# Drop rows where 'Finalists' is NaN
df_clean = df.dropna(subset=['Finalists'])

# Compute correlation with 'Finalists'
correlations = {
    'Swimsuit': df_clean['Swimsuit'].corr(df_clean['Finalists']),
    'Interview': df_clean['Interview'].corr(df_clean['Finalists']),
    'Evening Gown': df_clean['Evening Gown'].corr(df_clean['Finalists'])
}

# Check if any correlation has absolute value > 0.3 (considered significant)
significant_factors = []
for factor, corr in correlations.items():
    if abs(corr) > 0.3:
        significant_factors.append(factor)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")