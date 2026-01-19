import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'bp 2nd comp (˚c)' and 'bp 3rd comp (˚c)' columns by removing 'degree' and converting to float
def clean_temp(x):
    if isinstance(x, str):
        x = x.replace('degree', '').strip()
        return float(x) if x else np.nan
    return x

df['bp 2nd comp (˚c)'] = df['bp 2nd comp (˚c)'].apply(clean_temp)
df['bp 3rd comp (˚c)'] = df['bp 3rd comp (˚c)'].apply(clean_temp)

# Extract the last numeric value from '% wt 2nd' and '% wt 3rd'
def extract_weight(x):
    if isinstance(x, str):
        # Extract all numbers and take the last one
        nums = [float(i) for i in x.split() if i.replace('.', '', 1).isdigit()]
        return nums[-1] if nums else np.nan
    return x

df['% wt 2nd'] = df['% wt 2nd'].apply(extract_weight)
df['% wt 3rd'] = df['% wt 3rd'].apply(extract_weight)

# Convert 'bp azeo (˚c)' to float, removing any non-numeric parts
def clean_azeo(x):
    if isinstance(x, str):
        x = x.replace('degree', '').strip()
        return float(x) if x else np.nan
    return x

df['bp azeo (˚c)'] = df['bp azeo (˚c)'].apply(clean_azeo)

# Select relevant columns
columns_to_check = ['2nd component', 'bp 2nd comp (˚c)', '3rd component', 'bp 3rd comp (˚c)', '% wt 2nd', '% wt 3rd']
correlations = {}

for col in columns_to_check:
    if col == '2nd component' or col == '3rd component':
        # These are categorical; correlation not meaningful, skip
        correlations[col] = 0
    else:
        # Compute correlation with 'bp azeo (˚c)'
        corr = df[col].corr(df['bp azeo (˚c)'])
        correlations[col] = corr

# Find the variable with the highest absolute correlation
best_factor = max(correlations, key=lambda x: abs(correlations[x]))
print(f"Final Answer: {best_factor}")