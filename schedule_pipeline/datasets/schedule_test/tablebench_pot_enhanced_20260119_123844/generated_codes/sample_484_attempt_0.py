import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with invalid values
df.dropna(subset=['elevation (m)', 'prominence (m)'], inplace=True)

# Calculate mean and std for elevation and prominence
elev_mean, elev_std = df['elevation (m)'].mean(), df['elevation (m)'].std()
prom_mean, prom_std = df['prominence (m)'].mean(), df['prominence (m)'].std()

# Define threshold (2 standard deviations)
elev_threshold = elev_mean + 2 * elev_std
prom_threshold = prom_mean + 2 * prom_std

# Identify peaks with significant deviation
outliers = df[(df['elevation (m)'] > elev_threshold) | (df['prominence (m)'] > prom_threshold)]

# Extract peak names
outlier_peaks = outliers['peak'].tolist()

print(f"Final Answer: {', '.join(outlier_peaks)}")