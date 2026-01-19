import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric, handling any parsing issues
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Remove rows with missing values
df = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Define IQR method for outlier detection
def detect_outliers(df, column, lower_bound=-np.inf, upper_bound=np.inf):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_fence) | (df[column] > upper_fence)]
    return outliers

# Detect outliers in elevation and prominence
elevation_outliers = detect_outliers(df, 'elevation (m)')
prominence_outliers = detect_outliers(df, 'prominence (m)')

# Combine unique mountain names from both outlier sets
outlier_names = set()
for _, row in elevation_outliers.iterrows():
    outlier_names.add(row['peak'])
for _, row in prominence_outliers.iterrows():
    outlier_names.add(row['peak'])

# Convert to list and print
outlier_list = list(outlier_names)
print(f"Final Answer: {', '.join(outlier_list)}")