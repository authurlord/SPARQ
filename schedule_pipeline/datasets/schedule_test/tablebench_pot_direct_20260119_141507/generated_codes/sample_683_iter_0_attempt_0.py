import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'total fertility rate' and 'natural growth' to numeric, handling 'na' as NaN
df['total fertility rate'] = pd.to_numeric(df['total fertility rate'], errors='coerce')
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')

# Drop rows where either column is NaN
df_clean = df.dropna(subset=['total fertility rate', 'natural growth'])

# Compute correlation coefficient
correlation = df_clean['total fertility rate'].corr(df_clean['natural growth'])

print(f"Final Answer: {correlation:.3f}")