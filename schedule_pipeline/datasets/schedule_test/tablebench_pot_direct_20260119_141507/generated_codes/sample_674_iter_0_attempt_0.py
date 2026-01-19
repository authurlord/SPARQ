import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'poles' and 'wins' to numeric, coercing errors to NaN
df['poles'] = pd.to_numeric(df['poles'], errors='coerce')
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')

# Drop rows with missing values in 'poles' or 'wins'
df_clean = df.dropna(subset=['poles', 'wins'])

# Calculate the correlation coefficient
correlation = df_clean['poles'].corr(df_clean['wins'])

print(f"Final Answer: {correlation:.3f}")