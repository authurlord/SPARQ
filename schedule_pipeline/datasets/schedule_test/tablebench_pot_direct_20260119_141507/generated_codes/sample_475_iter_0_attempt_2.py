import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling any non-numeric or missing values
df['2011 census'] = pd.to_numeric(df['2011 census'], errors='coerce')
df['land area (km square)'] = pd.to_numeric(df['land area (km square)'], errors='coerce')
df['density (pop / km square)'] = pd.to_numeric(df['density (pop / km square)'], errors='coerce')

# Remove rows with NaN due to conversion issues
df = df.dropna()

# Identify outliers using IQR method for '2011 census' and 'land area'
def find_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# Find outliers in 2011 census and land area
outliers_census = find_outliers_iqr(df['2011 census'])
outliers_area = find_outliers_iqr(df['land area (km square)'])

# Check for very high or very low density
outliers_density = find_outliers_iqr(df['density (pop / km square)'])

# Also check for invalid land area (negative)
invalid_area = df[df['land area (km square)'] < 0]

# Combine findings
unusual_entries = []
if not outliers_census.empty:
    unusual_entries.extend(outliers_census.index.tolist())
if not outliers_area.empty:
    unusual_entries.extend(outliers_area.index.tolist())
if not outliers_density.empty:
    unusual_entries.extend(outliers_density.index.tolist())
if not invalid_area.empty:
    unusual_entries.extend(invalid_area.index.tolist())

# Get the names of the regions corresponding to these indices
unusual_names = df.loc[unusual_entries, 'name'].drop_duplicates().tolist()

# Final answer: list of unusual region names
print(f"Final Answer: {', '.join(unusual_names)}")