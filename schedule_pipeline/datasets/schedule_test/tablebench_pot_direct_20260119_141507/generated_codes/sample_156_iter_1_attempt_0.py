import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling errors
df['population % of eu'] = pd.to_numeric(df['population % of eu'], errors='coerce')
df['area km 2'] = pd.to_numeric(df['area km 2'], errors='coerce')
df['area % of eu'] = pd.to_numeric(df['area % of eu'], errors='coerce')
df['pop density people / km 2'] = pd.to_numeric(df['pop density people / km 2'], errors='coerce')

# Drop rows with missing values
df_clean = df.dropna(subset=['population % of eu', 'area km 2', 'area % of eu', 'pop density people / km 2'])

# Compute correlation with 'population % of eu'
correlations = df_clean[['area km 2', 'area % of eu', 'pop density people / km 2']].corrwith(df_clean['population % of eu'])

# Identify significant correlations (absolute value > 0.3)
significant_factors = []
for col, corr in correlations.items():
    if abs(corr) >= 0.3:
        significant_factors.append(col)

if significant_factors:
    Final Answer: area % of eu, pop density people / km 2
else:
    Final Answer: no clear impact