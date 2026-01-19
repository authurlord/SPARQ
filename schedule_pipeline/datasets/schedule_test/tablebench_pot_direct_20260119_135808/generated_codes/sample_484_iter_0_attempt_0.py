import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate mean and std for elevation and prominence
elev_mean, elev_std = df['elevation (m)'].mean(), df['elevation (m)'].std()
prom_mean, prom_std = df['prominence (m)'].mean(), df['prominence (m)'].std()

# Identify outliers (more than 2 std away)
elev_outliers = df[(df['elevation (m)'] > elev_mean + 2*elev_std) | (df['elevation (m)'] < elev_mean - 2*elev_std)]
prom_outliers = df[(df['prominence (m)'] > prom_mean + 2*prom_std) | (df['prominence (m)'] < prom_mean - 2*prom_std)]

# Combine outlier peaks
outlier_peaks = set(elev_outliers['peak']).union(set(prom_outliers['peak']))

print(f"Final Answer: {', '.join(outlier_peaks)}")