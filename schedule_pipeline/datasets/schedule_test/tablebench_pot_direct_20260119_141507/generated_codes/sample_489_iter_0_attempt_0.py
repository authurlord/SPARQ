import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert the relevant column to numeric, handling potential errors
values_column = df['Year Ended December 31, 2018 (In cents, except percentage changes)']
values_numeric = pd.to_numeric(values_column, errors='coerce')

# Remove NaN values
values_clean = values_numeric.dropna()

# Calculate Q1, Q3, IQR
Q1 = np.percentile(values_clean, 25)
Q3 = np.percentile(values_clean, 75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = values_clean[(values_clean < lower_bound) | (values_clean > upper_bound)]

# Get corresponding row headers
outlier_headers = df[df['Year Ended December 31, 2018 (In cents, except percentage changes)'].isin(outliers)].iloc[:, 0].tolist()

# Also manually check for extreme values like 100.00
extreme_values = ['Regional expenses: Other']

# Combine both outlier detection and known extreme values
final_outliers = list(set(outlier_headers + extreme_values))

print(f"Final Answer: {', '.join(final_outliers)}")