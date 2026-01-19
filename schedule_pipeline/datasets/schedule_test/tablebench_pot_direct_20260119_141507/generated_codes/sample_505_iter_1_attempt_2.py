import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert all number columns to numeric, coercing errors to NaN
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'], errors='coerce')
df['number of hurricanes'] = pd.to_numeric(df['number of hurricanes'], errors='coerce')
df['number of major hurricanes'] = pd.to_numeric(df['number of major hurricanes'], errors='coerce')

# Drop rows where tropical storms is NaN (invalid)
df_clean = df.dropna(subset=['number of tropical storms', 'number of hurricanes', 'number of major hurricanes'])

# Calculate HSI for each year
df_clean['hs_index'] = (df_clean['number of hurricanes'] + df_clean['number of major hurricanes']) / df_clean['number of tropical storms']

# Find the year with the highest HSI
max_hsi_year = df_clean.loc[df_clean['hs_index'].idxmax(), 'year']

print(f"Final Answer: {max_hsi_year}")