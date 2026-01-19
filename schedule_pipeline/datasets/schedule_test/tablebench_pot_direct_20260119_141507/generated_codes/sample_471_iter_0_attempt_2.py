import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')
df['col (m)'] = pd.to_numeric(df['col (m)'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Calculate mean and std for elevation and prominence
mean_elev = df['elevation (m)'].mean()
std_elev = df['elevation (m)'].std()

mean_prominence = df['prominence (m)'].mean()
std_prominence = df['prominence (m)'].std()

# Identify outliers in elevation and prominence (more than 2 std from mean)
outliers_elev = df[(df['elevation (m)'] > mean_elev + 2 * std_elev) | (df['elevation (m)'] < mean_elev - 2 * std_elev)]
outliers_prominence = df[(df['prominence (m)'] > mean_prominence + 2 * std_prominence) | (df['prominence (m)'] < mean_prominence - 2 * std_prominence)]

# Check for unusually high col values
outliers_col = df[df['col (m)'] > 2000]  # Since most col values are under 2000, anything above 2000 is notable

# Combine all outlier observations
outlier_peaks = []
if not outliers_elev.empty:
    outlier_peaks.extend(outliers_elev['peak'].tolist())
if not outliers_prominence.empty:
    outlier_peaks.extend(outliers_prominence['peak'].tolist())
if not outliers_col.empty:
    outlier_peaks.extend(outliers_col['peak'].tolist())

# Remove duplicates
outlier_peaks = list(set(outlier_peaks))

print(f"Final Answer: {', '.join(outlier_peaks)}")