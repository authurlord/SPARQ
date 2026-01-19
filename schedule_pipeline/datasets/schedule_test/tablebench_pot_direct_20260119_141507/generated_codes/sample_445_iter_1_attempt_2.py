import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns
columns = ['eccentricity', 'inclination (deg) (to moon equator)', 'periselene (km)', 'aposelene (km)']
data = df[columns].copy()

# Function to detect outliers using IQR method
def detect_outliers_iqr(df):
    outliers = []
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers.extend(outliers_in_col.index.tolist())
    # Remove duplicates
    unique_outliers = list(set(outliers))
    return unique_outliers

# Detect outlier indices
outlier_indices = detect_outliers_iqr(data)

# Get the original row indices (from the original DataFrame)
# Convert to readable format: e.g., "March 15, 2005, 12:00:00.0"
original_indices = [str(df.iloc[i]['epoch (utc)']) for i in outlier_indices]

# Final answer: list of epoch times where anomalies occur
print(f"Final Answer: {', '.join(original_indices)}")