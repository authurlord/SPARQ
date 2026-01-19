import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Remove rows with missing values
df = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Calculate z-scores for elevation and prominence
z_elevation = np.abs((df['elevation (m)'] - df['elevation (m)'].mean()) / df['elevation (m)'].std())
z_prominence = np.abs((df['prominence (m)'] - df['prominence (m)'].mean()) / df['prominence (m)'].std())

# Identify outliers (z-score > 2)
outliers_elev = z_elevation > 2
outliers_prominence = z_prominence > 2

# Find peaks that are outliers in either elevation or prominence
outlier_peaks = df[(outliers_elev) | (outliers_prominence)]
outlier_names = outlier_peaks['peak'].tolist()

print(f"Final Answer: {', '.join(outlier_names)}")