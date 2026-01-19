import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Remove rows with NaN due to conversion errors
df = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Calculate mean and std for elevation and prominence
mean_elev = df['elevation (m)'].mean()
std_elev = df['elevation (m)'].std()

mean_prom = df['prominence (m)'].mean()
std_prom = df['prominence (m)'].std()

# Identify outliers using 2 standard deviations rule
outliers_elev = df[(df['elevation (m)'] > mean_elev + 2 * std_elev) | (df['elevation (m)'] < mean_elev - 2 * std_elev)]
outliers_prom = df[(df['prominence (m)'] > mean_prom + 2 * std_prom) | (df['prominence (m)'] < mean_prom - 2 * std_prom)]

# Combine and get unique peak names
outlier_peaks = set()
for idx, row in outliers_elev.iterrows():
    outlier_peaks.add(row['peak'])
for idx, row in outliers_prom.iterrows():
    outlier_peaks.add(row['peak'])

# Return the peak names
outlier_names = list(outlier_peaks)
print(f"Final Answer: {', '.join(outlier_names)}")