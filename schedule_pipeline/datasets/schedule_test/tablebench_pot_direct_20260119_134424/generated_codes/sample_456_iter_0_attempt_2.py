import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['elevation (m)', 'prominence (m)'], inplace=True)

# Calculate mean and std for elevation and prominence
elev_mean, elev_std = df['elevation (m)'].mean(), df['elevation (m)'].std()
prom_mean, prom_std = df['prominence (m)'].mean(), df['prominence (m)'].std()

# Define threshold (2 standard deviations)
elev_outliers = df[(df['elevation (m)'] > elev_mean + 2 * elev_std) | (df['elevation (m)'] < elev_mean - 2 * elev_std)]
prom_outliers = df[(df['prominence (m)'] > prom_mean + 2 * prom_std) | (df['prominence (m)'] < prom_mean - 2 * prom_std)]

# Combine outlier peaks
outliers = pd.concat([elev_outliers, prom_outliers]).drop_duplicates()

# Extract peak names
outlier_peaks = outliers['peak'].tolist()

print(f"Final Answer: {', '.join(outlier_peaks)}")