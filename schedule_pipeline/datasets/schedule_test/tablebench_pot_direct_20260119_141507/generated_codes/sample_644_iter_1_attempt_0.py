import pandas as pd
import numpy as np

# Load the table
df = pd.read_csv('table.csv')

# Extract boiling point (degree) values (from row "boiling point (degree)")
boiling_point = df.loc[df['physical property'] == 'boiling point (degree)', 'helium':].iloc[0].dropna().astype(float)

# Extract critical temperature (k) values (from row "critical temperature (k)")
critical_temp = df.loc[df['physical property'] == 'critical temperature (k)', 'helium':].iloc[0].dropna().astype(float)

# Calculate the correlation coefficient
correlation = np.corrcoef(boiling_point, critical_temp)[0, 1]

print(f"Final Answer: {correlation:.3f}")