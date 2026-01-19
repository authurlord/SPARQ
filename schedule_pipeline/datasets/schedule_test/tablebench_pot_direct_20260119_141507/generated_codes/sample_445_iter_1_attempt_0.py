import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select only numeric columns for outlier detection
numeric_columns = ['periselene (km)', 'aposelene (km)', 'eccentricity', 'inclination (deg) (to moon equator)']
df_numeric = df[numeric_columns].copy()

# Function to detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Apply outlier detection to each column
outlier_mask = pd.Series([False] * len(df))
for col in numeric_columns:
    outlier_mask |= detect_outliers_iqr(df_numeric, col)

# Get the rows where any outlier exists
unusual_rows = df[outlier_mask].index.tolist()

# Print the rows (we'll show the full row data for clarity)
print("Unusual data points (by index):")
for idx in unusual_rows:
    print(df.loc[idx].to_dict())

Final Answer: 1, 2, 3, 4