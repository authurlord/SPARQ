import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')
df['col (m)'] = pd.to_numeric(df['col (m)'], errors='coerce')

# Identify outliers using thresholds
outliers = []

# Check for high col values (e.g., above 90% percentile)
col_threshold = df['col (m)'].quantile(0.95)
high_col_peaks = df[df['col (m)'] > col_threshold]

# Check for low elevation with high prominence or col
low_elevation = df[(df['elevation (m)'] < df['elevation (m)'].quantile(0.1)) & 
                   (df['prominence (m)'] > df['prominence (m)'].quantile(0.5))]

# Combine results
outlier_peaks = high_col_peaks[['peak']].values.tolist()
outlier_peaks.extend(low_elevation[['peak']].values.tolist())

# Remove duplicates and clean
unique_outliers = list(set([item[0] for item in outlier_peaks]))

print(f"Final Answer: {', '.join(unique_outliers)}")