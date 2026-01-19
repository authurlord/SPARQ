import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns
columns = ['eccentricity', 'inclination (deg) (to moon equator)', 'periselene (km)', 'aposelene (km)']
data = df[columns].copy()

# Function to detect outliers using IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Apply outlier detection to each column
outlier_mask = data.apply(detect_outliers_iqr).any(axis=1)

# Get the rows with unusual patterns
unusual_rows = df[outlier_mask].index.tolist()

# Print the indices or the actual data points (we'll print the index for clarity)
print(f"Final Answer: {unusual_rows}")