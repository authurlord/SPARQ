import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert 'bp azeo (˚c)' to numeric, removing 'degree' from string
df['bp azeo (˚c)'] = df['bp azeo (˚c)'].str.replace('degree', '').str.strip().astype(float)

# Extract numeric values from '% wt 1st', '% wt 2nd', '% wt 3rd'
def extract_numeric(s):
    # Extract first number in string
    try:
        return float(s.split()[0])
    except:
        return np.nan

# Apply to % wt 2nd and % wt 3rd
df['% wt 2nd'] = df['% wt 2nd'].apply(extract_numeric)
df['% wt 3rd'] = df['% wt 3rd'].apply(extract_numeric)

# Drop rows with NaN in key columns
df = df.dropna(subset=['bp azeo (˚c)', 'bp 2nd comp (˚c)', 'bp 3rd comp (˚c)', '% wt 2nd', '% wt 3rd'])

# Compute correlation with 'bp azeo (˚c)'
correlations = {
    'bp 2nd comp (˚c)': df['bp 2nd comp (˚c)'].corr(df['bp azeo (˚c)']),
    'bp 3rd comp (˚c)': df['bp 3rd comp (˚c)'].corr(df['bp azeo (˚c)']),
    '% wt 2nd': df['% wt 2nd'].corr(df['bp azeo (˚c)']),
    '% wt 3rd': df['% wt 3rd'].corr(df['bp azeo (˚c)'])
}

# Find the factor with the highest absolute correlation
max_corr = max(correlations, key=lambda x: abs(correlations[x]))
print(f"Final Answer: {max_corr}")