import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert all values to numeric, handling any potential errors
df = df.apply(pd.to_numeric, errors='coerce')

# Filter out rows where 2011 value is missing
df_filtered = df.dropna(subset=['2011'])

# Calculate the increase from 2007 to 2011
df_filtered['increase'] = df_filtered['2011'] - df_filtered['2007']

# Find the school with the maximum increase
max_increase_school = df_filtered.loc[df_filtered['increase'].idxmax(), 'School']

print(f"Final Answer: {max_increase_school}")