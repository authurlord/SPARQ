import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert all relevant columns to numeric (handling any non-numeric values like 'n/a')
for col in df.columns:
    if col not in ['tourism receipts (2011) (millions of us)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN due to conversion
df = df.dropna()

# Select the target variable
target = 'tourism receipts (2011) (millions of us)'
correlations = {}

# Compute correlation with target for each factor
for col in df.columns:
    if col != target:
        corr = df[col].corr(df[target])
        correlations[col] = corr

# Find factors with absolute correlation > 0.5
significant_factors = [col for col, corr in correlations.items() if abs(corr) > 0.5]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")