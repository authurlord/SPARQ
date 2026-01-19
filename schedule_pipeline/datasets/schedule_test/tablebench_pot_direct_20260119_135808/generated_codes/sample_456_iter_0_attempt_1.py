import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric, handling potential non-numeric entries
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with missing values after conversion
df.dropna(subset=['elevation (m)', 'prominence (m)'], inplace=True)

# Calculate mean and std for elevation and prominence
elevation_mean = df['elevation (m)'].mean()
elevation_std = df['elevation (m)'].std()
prominence_mean = df['prominence (m)'].mean()
prominence_std = df['prominence (m)'].std()

# Define threshold for abnormal values (more than 2 std away)
elevation_outliers = df[(df['elevation (m)'] > elevation_mean + 2 * elevation_std) | 
                         (df['elevation (m)'] < elevation_mean - 2 * elevation_std)]
prominence_outliers = df[(df['prominence (m)'] > prominence_mean + 2 * prominence_std) | 
                          (df['prominence (m)'] < prominence_mean - 2 * prominence_std)]

# Combine outlier peaks
abnormal_peaks = pd.concat([elevation_outliers, prominence_outliers]).drop_duplicates()

# Extract peak names
abnormal_peak_names = abnormal_peaks['peak'].tolist()

print(f"Final Answer: {', '.join(abnormal_peak_names)}")