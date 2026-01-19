import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert numeric columns to float
numeric_cols = ['elevation (m)', 'prominence (m)', 'col (m)']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Function to detect outliers using IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Identify outlier rows across all numeric columns
outlier_rows = df.copy()
for col in numeric_cols:
    outlier_mask = detect_outliers_iqr(df[col])
    outlier_rows[f'{col}_outlier'] = outlier_mask

# Find any row where at least one column has an outlier
final_outliers = outlier_rows[outlier_rows[[f'{col}_outlier' for col in numeric_cols]].any(axis=1)]

# Print the rows with outliers
print(final_outliers[['peak', 'country', 'island', 'elevation (m)', 'prominence (m)', 'col (m)']])