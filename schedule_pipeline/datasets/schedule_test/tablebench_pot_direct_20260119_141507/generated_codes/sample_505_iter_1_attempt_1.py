import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, coercing errors to NaN
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'], errors='coerce')
df['number of hurricanes'] = pd.to_numeric(df['number of hurricanes'], errors='coerce')
df['number of major hurricanes'] = pd.to_numeric(df['number of major hurricanes'], errors='coerce')

# Calculate HSI: (hurricanes + major hurricanes) / tropical storms
df['hsi'] = (df['number of hurricanes'] + df['number of major hurricanes']) / df['number of tropical storms']

# Find the year with the highest HSI, skipping rows where tropical storms is 0 or NaN
valid_years = df.dropna(subset=['hsi'])
max_hsi_year = valid_years.loc[valid_years['hsi'].idxmax(), 'year']

print(f"Final Answer: {max_hsi_year}")