import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert the columns to numeric (replace commas and handle strings)
for col in df.columns:
    if col != 'Year':
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# Extract the relevant columns (Year_1, Year_2, Year_1) for analysis
columns_to_check = ['Year_1', 'Year_2', '-_1', '-_2']
outlier_years = []

for col in columns_to_check:
    if col not in df.columns:
        continue
    data = df[col].dropna()
    if len(data) < 2:
        continue
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    if not outliers.empty:
        # Get the corresponding year from the first column (Year)
        year_col = df['Year']
        outlier_years.extend(year_col[df[col].isin(outliers)].dropna().tolist())

# Remove duplicates and sort
unique_outlier_years = sorted(set(outlier_years))

print(f"Final Answer: {', '.join(map(str, unique_outlier_years))}")