import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['elevation (m)', 'prominence (m)'], inplace=True)

# Function to detect outliers using IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Detect outliers in elevation and prominence
elevation_outliers = detect_outliers_iqr(df['elevation (m)'])
prominence_outliers = detect_outliers_iqr(df['prominence (m)'])

# Combine outlier flags
outliers = elevation_outliers | prominence_outliers

# Get the names of peaks with abnormal values
abnormal_peaks = df[outliers]['peak'].tolist()

print(f"Final Answer: {', '.join(abnormal_peaks)}")