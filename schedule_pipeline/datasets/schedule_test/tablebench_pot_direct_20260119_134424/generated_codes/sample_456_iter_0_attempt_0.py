import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with invalid values
df.dropna(subset=['elevation (m)', 'prominence (m)'], inplace=True)

# Calculate mean and std for elevation and prominence
elevation_mean = df['elevation (m)'].mean()
elevation_std = df['elevation (m)'].std()

prominence_mean = df['prominence (m)'].mean()
prominence_std = df['prominence (m)'].std()

# Identify outliers (more than 2 std away)
elevation_outliers = df[abs(df['elevation (m)'] - elevation_mean) > 2 * elevation_std]
prominence_outliers = df[abs(df['prominence (m)'] - prominence_mean) > 2 * prominence_std]

# Combine outlier peaks
outliers = pd.concat([elevation_outliers, prominence_outliers]).drop_duplicates()

# Extract peak names
outlier_peaks = outliers['peak'].tolist()

print(f"Final Answer: {', '.join(outlier_peaks)}")