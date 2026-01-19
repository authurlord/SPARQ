import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'sunshine (hrs / year)' and 'storms (days / year)' columns by removing spaces
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '')
df['storms (days / year)'] = df['storms (days / year)'].str.replace(' ', '')

# Convert to numeric
df['sunshine (hrs / year)'] = pd.to_numeric(df['sunshine (hrs / year)'], errors='coerce')
df['storms (days / year)'] = pd.to_numeric(df['storms (days / year)'], errors='coerce')

# Calculate correlation
correlation = df['sunshine (hrs / year)'].corr(df['storms (days / year)'])

print(f"Final Answer: {correlation:.3f}")