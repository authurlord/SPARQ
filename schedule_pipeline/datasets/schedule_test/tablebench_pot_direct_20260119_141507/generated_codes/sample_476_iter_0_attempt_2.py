import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert all values to numeric (handle commas and spaces)
def parse_numeric(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return x

# Apply parsing to all columns
for col in df.columns:
    df[col] = df[col].apply(parse_numeric)

# Function to detect outliers using IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.index.tolist()

# Identify outlier years in each column
outlier_years = {}
for col in df.columns:
    if col != '-':  # Skip empty or non-numeric columns
        outlier_years[col] = detect_outliers_iqr(df[col])

# Find years that appear as outliers in any column
all_outlier_years = set()
for col, indices in outlier_years.items():
    all_outlier_years.update(indices)

# Print the years with significant deviations
print(f"Final Answer: {', '.join(map(str, sorted(all_outlier_years)))}")