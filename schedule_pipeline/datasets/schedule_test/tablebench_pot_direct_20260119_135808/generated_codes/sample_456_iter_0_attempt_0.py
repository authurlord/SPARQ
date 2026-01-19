import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['elevation (m)', 'prominence (m)'], inplace=True)

# Calculate Z-scores for elevation and prominence
elevation_z = np.abs((df['elevation (m)'] - df['elevation (m)'].mean()) / df['elevation (m)'].std())
prominence_z = np.abs((df['prominence (m)'] - df['prominence (m)'].mean()) / df['prominence (m)'].std())

# Identify outliers (Z-score > 2)
outliers = df[(elevation_z > 2) | (prominence_z > 2)]

# Extract peak names
outlier_peaks = outliers['peak'].tolist()

print(f"Final Answer: {', '.join(outlier_peaks)}")