import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric, coercing errors to NaN
df['Swimsuit'] = pd.to_numeric(df['Swimsuit'], errors='coerce')
df['Interview'] = pd.to_numeric(df['Interview'], errors='coerce')
df['Evening Gown'] = pd.to_numeric(df['Evening Gown'], errors='coerce')
df['Finalists'] = pd.to_numeric(df['Finalists'], errors='coerce')

# Drop rows with missing Finalists (since we need them for correlation)
df_clean = df.dropna(subset=['Finalists'])

# Calculate correlation between each factor and Finalists
correlations = {
    'Swimsuit': df_clean['Swimsuit'].corr(df_clean['Finalists']),
    'Interview': df_clean['Interview'].corr(df_clean['Finalists']),
    'Evening Gown': df_clean['Evening Gown'].corr(df_clean['Finalists'])
}

# Find the factor with the highest absolute correlation
max_corr_factor = max(correlations, key=correlations.get)
max_corr_value = abs(correlations[max_corr_factor])

# If no factor has a meaningful correlation (e.g., > 0.5), return 'no clear impact'
if max_corr_value < 0.5:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {max_corr_factor}")