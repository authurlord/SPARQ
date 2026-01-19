import pandas as pd
import numpy as np

# Load the table
df = pd.read_csv('table.csv')

# Clean the 'bp azeo (˚c)' column by removing 'degree' and converting to float
def clean_bp_azeo(x):
    if isinstance(x, str):
        x = x.replace('degree', '').strip()
        if x.endswith('degree'):
            x = x[:-7].strip()
        try:
            return float(x)
        except ValueError:
            return np.nan
    return np.nan

df['bp azeo (˚c)'] = df['bp azeo (˚c)'].apply(clean_bp_azeo)

# Convert other columns to numeric where possible
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'], errors='coerce')
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'], errors='coerce')
df['% wt 2nd'] = pd.to_numeric(df['% wt 2nd'].str.replace('u', '', regex=False), errors='coerce')
df['% wt 3rd'] = pd.to_numeric(df['% wt 3rd'].str.replace('u', '', regex=False), errors='coerce')

# Drop rows with NaN in 'bp azeo (˚c)' for correlation analysis
df_clean = df.dropna(subset=['bp azeo (˚c)'])

# Compute correlation with 'bp azeo (˚c)'
correlations = df_clean[['bp 2nd comp (˚c)', 'bp 3rd comp (˚c)', '% wt 2nd', '% wt 3rd']].corrwith(df_clean['bp azeo (˚c)'])
correlations = correlations.abs().sort_values(ascending=False)

# Get the top influencing factor
main_factor = correlations.index[0]
print(f"Final Answer: {main_factor}")