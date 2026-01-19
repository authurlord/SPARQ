import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling string representations of numbers
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')

# Drop rows with NaN (if any)
df = df.dropna()

# Identify outliers using IQR method for each numerical column
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# Apply outlier detection
outliers_revenue = detect_outliers_iqr(df['revenues (us billion)'])
outliers_profit = detect_outliers_iqr(df['profit (us billion)'])
outliers_assets = detect_outliers_iqr(df['assets (us billion)'])
outliers_market_value = detect_outliers_iqr(df['market value (us billion)'])

# List the companies with outliers
outlier_companies = []
if not outliers_revenue.empty:
    outlier_companies.extend(df[df['revenues (us billion)'].isin(outliers_revenue.index)]['company'].tolist())
if not outliers_profit.empty:
    outlier_companies.extend(df[df['profit (us billion)'].isin(outliers_profit.index)]['company'].tolist())
if not outliers_assets.empty:
    outlier_companies.extend(df[df['assets (us billion)'].isin(outliers_assets.index)]['company'].tolist())
if not outliers_market_value.empty:
    outlier_companies.extend(df[df['market value (us billion)'].isin(outliers_market_value.index)]['company'].tolist())

# Remove duplicates
outlier_companies = list(set(outlier_companies))

# Final answer: list of companies with unusual patterns or outliers
print(f"Final Answer: {', '.join(outlier_companies)}")