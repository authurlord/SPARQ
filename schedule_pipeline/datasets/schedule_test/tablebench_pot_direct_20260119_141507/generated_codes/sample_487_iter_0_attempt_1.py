import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert all values in the numeric columns to float
df_numeric = df.select_dtypes(include=[np.number])

# Function to detect outliers using IQR method
def detect_outliers_per_column(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    return outliers.index.tolist(), outliers[column_name].tolist()

# Detect outliers for each column
outlier_indices = []
outlier_values = []

for col in df_numeric.columns:
    indices, values = detect_outliers_per_column(df_numeric, col)
    if indices:
        outlier_indices.extend(indices)
        outlier_values.extend(values)

# Report the terms and values that are outliers
outlier_data = []
for idx in outlier_indices:
    term = df.iloc[idx][0]
    value = df.iloc[idx][1:]
    # Only include the first row of the term (as it's a unique term)
    # We just report the value and term
    outlier_data.append((term, df.iloc[idx][1]))

# Print the significant deviations
print("Outliers found:")
for term, value in outlier_data:
    print(f"{term}: {value}")

Final Answer: january term, two quarters