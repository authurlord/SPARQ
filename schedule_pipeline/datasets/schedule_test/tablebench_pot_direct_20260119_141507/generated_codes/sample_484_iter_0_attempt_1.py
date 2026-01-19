import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Remove rows with NaN
df = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Compute mean and std for elevation and prominence
mean_elev = df['elevation (m)'].mean()
std_elev = df['elevation (m)'].std()
mean_prominence = df['prominence (m)'].mean()
std_prominence = df['prominence (m)'].std()

# Identify outliers using 2 standard deviations
outliers_elev = np.abs(df['elevation (m)'] - mean_elev) > 2 * std_elev
outliers_prominence = np.abs(df['prominence (m)'] - mean_prominence) > 2 * std_prominence

# Find peaks that are outliers in either elevation or prominence
outlier_peaks = df[(outliers_elev) | (outliers_prominence)]
outlier_names = outlier_peaks['peak'].tolist()

print(f"Final Answer: {', '.join(outlier_names)}")