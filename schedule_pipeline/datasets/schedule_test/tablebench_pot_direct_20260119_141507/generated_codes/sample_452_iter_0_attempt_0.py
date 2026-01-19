import pandas as pd
import numpy as np
df = pd.read_csv('table.csv')
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'].str.replace('-', ''), errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'].str.replace('-', ''), errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
outliers_revenue = detect_outliers_iqr(df['revenues (us billion)'])
outliers_profit = detect_outliers_iqr(df['profit (us billion)'])
outliers_assets = detect_outliers_iqr(df['assets (us billion)'])
outlier_companies = []
anomalies = []
for idx, row in df.iterrows():
final_patterns = [