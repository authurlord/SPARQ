import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate z-scores for elevation and prominence
elevation_z = np.abs((df['elevation (m)'] - df['elevation (m)'].mean()) / df['elevation (m)'].std())
prominence_z = np.abs((df['prominence (m)'] - df['prominence (m)'].mean()) / df['prominence (m)'].std())

# Identify peaks with z-score > 2 in either column
outliers = df[(elevation_z > 2) | (prominence_z > 2)]

# Extract peak names
outlier_peaks = outliers['peak'].tolist()
print(f"Final Answer: {', '.join(outlier_peaks)}")