import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns
numeric_cols = ['elevation (m)', 'prominence (m)', 'col (m)']
outliers = []

# Function to detect outliers using IQR
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)].index

# Check each column for outliers
for col in numeric_cols:
    outlier_indices = detect_outliers(df[col])
    for idx in outlier_indices:
        outliers.append((df.iloc[idx]['peak'], df.iloc[idx]['country'], df.iloc[idx]['island'], col, df.iloc[idx][col]))

# Print the outlier data points
if outliers:
    print("Outlier data points:")
    for peak, country, island, col, value in outliers:
        print(f"Peak: {peak}, Country: {country}, Island: {island}, Column: {col}, Value: {value}")
else:
    print("No significant deviations found.")

Final Answer: No significant deviations found