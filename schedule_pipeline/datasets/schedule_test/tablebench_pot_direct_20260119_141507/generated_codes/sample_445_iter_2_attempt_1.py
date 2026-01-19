import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'], errors='coerce')
df['aposelene (km)'] = pd.to_numeric(df['aposelene (km)'], errors='coerce')
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['inclination (deg) (to moon equator)'] = pd.to_numeric(df['inclination (deg) (to moon equator)'], errors='coerce')

# Remove any rows with NaN after conversion
df = df.dropna()

# Calculate z-scores for each column
def detect_outliers_zscore(series, threshold=2):
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return False
    z_scores = np.abs((series - mean_val) / std_val)
    return z_scores > threshold

outlier_rows = []

for idx, row in df.iterrows():
    if (detect_outliers_zscore(row['eccentricity']) or 
        detect_outliers_zscore(row['inclination (deg) (to moon equator)']) or 
        detect_outliers_zscore(row['periselene (km)']) or 
        detect_outliers_zscore(row['aposelene (km)'])):
        outlier_rows.append(row.to_dict())

# If no outliers found, print message
if not outlier_rows:
    print("Final Answer: No unusual patterns detected")
else:
    # Print the rows with unusual patterns
    for row in outlier_rows:
        print(f"Final Answer: {row['epoch (utc)']}, periselene: {row['periselene (km)']}, aposelene: {row['aposelene (km)']}, eccentricity: {row['eccentricity']}, inclination: {row['inclination (deg) (to moon equator)']}")