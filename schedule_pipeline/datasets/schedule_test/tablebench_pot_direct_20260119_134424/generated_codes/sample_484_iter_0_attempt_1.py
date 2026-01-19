import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate IQR for elevation and prominence
def find_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]

# Find outliers in elevation and prominence
elevation_outliers = find_outliers(df['elevation (m)'])
prominence_outliers = find_outliers(df['prominence (m)'])

# Get peak names with outliers
outlier_peaks = set(elevation_outliers.index).union(prominence_outliers.index)
outlier_peak_names = df.loc[outlier_peaks, 'peak'].tolist()

print(f"Final Answer: {', '.join(outlier_peak_names)}")