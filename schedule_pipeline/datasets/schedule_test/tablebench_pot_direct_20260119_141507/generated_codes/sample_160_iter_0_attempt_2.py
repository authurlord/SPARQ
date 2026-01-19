import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric (handle any non-numeric entries)
df['Shared Titles'] = pd.to_numeric(df['Shared Titles'], errors='coerce')
df['Runners-Up'] = pd.to_numeric(df['Runners-Up'], errors='coerce')
df['Total Finals'] = pd.to_numeric(df['Total Finals'], errors='coerce')
df['Outright Titles'] = pd.to_numeric(df['Outright Titles'], errors='coerce')

# Drop rows with NaN due to conversion errors
df = df.dropna()

# Compute correlation with 'Outright Titles'
correlations = df[['Shared Titles', 'Runners-Up', 'Total Finals']].corrwith(df['Outright Titles'])

# Find the factor with the highest absolute correlation
max_corr = correlations.abs().idxmax()
max_corr_value = correlations.abs().max()

# If the max correlation is below a threshold (e.g., 0.3), consider no clear impact
if max_corr_value < 0.3:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {max_corr}")