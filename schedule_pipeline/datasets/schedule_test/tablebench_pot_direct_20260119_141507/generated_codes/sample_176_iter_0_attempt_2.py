import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'season' and 'tv season' to numeric (they are already strings like '1995-1996')
df['season'] = df['season'].astype(str)
df['tv season'] = df['tv season'].astype(str)

# Extract numeric parts from 'tv season' to get a numerical value (e.g., 1995-1996 -> 1995)
df['tv_year'] = df['tv season'].str.split('-').str[0].astype(int)

# Convert 'rank' to numeric
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Drop rows with NaN due to conversion
df = df.dropna()

# Compute correlation with 'viewers (in millions)'
correlations = df.corr()['viewers (in millions)'].drop('viewers (in millions)')

# Filter only significant correlations (|r| > 0.3)
significant_factors = [col for col in correlations.index if abs(correlations[col]) > 0.3]

if len(significant_factors) == 0:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {', '.join(significant_factors)}")