import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert the numeric columns to float and handle missing values
df_numeric = df[['School', '2007', '2008', '2009', '2010', '2011']].apply(pd.to_numeric, errors='coerce')

# Drop rows where 2007 or 2011 is missing
df_clean = df_numeric.dropna(subset=['2007', '2011'])

# Calculate the increase from 2007 to 2011
df_clean['increase'] = df_clean['2011'] - df_clean['2007']

# Find the school with the maximum increase
max_increase_school = df_clean.loc[df_clean['increase'].idxmax(), 'School']

print(f"Final Answer: {max_increase_school}")