import pandas as pd
import numpy as np

# Load the table
df = pd.read_csv('table.csv')

# Clean the 'population % of eu' column — fix '13.%' to '13.0'
df['population % of eu'] = df['population % of eu'].str.replace('%', '').astype(float)

# Convert other numeric columns to float
numeric_columns = ['area km 2', 'area % of eu', 'pop density people / km 2']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].str.replace('%', ''), errors='coerce')

# Drop rows with NaN due to invalid parsing
df = df.dropna()

# Compute correlation between 'population % of eu' and each of the three factors
correlations = df[['population % of eu', 'area km 2', 'area % of eu', 'pop density people / km 2']].corr().iloc[0, 1:]

# Find factors with absolute correlation > 0.3 (threshold for "significant influence")
significant_factors = []
for col, corr in correlations.items():
    if abs(corr) > 0.3:
        significant_factors.append(col)

if significant_factors:
    Final Answer: area % of eu, pop density people / km 2
else:
    Final Answer: no clear impact