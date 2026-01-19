import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate mean and std for elevation and prominence
elevation_mean = df['elevation (m)'].mean()
elevation_std = df['elevation (m)'].std()
prominence_mean = df['prominence (m)'].mean()
prominence_std = df['prominence (m)'].std()

# Identify outliers (more than 2 std away)
outliers = df[
    (df['elevation (m)'] > elevation_mean + 2 * elevation_std) |
    (df['elevation (m)'] < elevation_mean - 2 * elevation_std) |
    (df['prominence (m)'] > prominence_mean + 2 * prominence_std) |
    (df['prominence (m)'] < prominence_mean - 2 * prominence_std)
]

# Extract peak names
outlier_peaks = outliers['peak'].tolist()
print(f"Final Answer: {', '.join(outlier_peaks)}")