import pandas as pd
import re

# Load the data
df = pd.read_csv('table.csv', index_col='Unnamed: 0')

# Extract the "2001 general" column (index 6)
column_name = '2001 general'
values = df[column_name].dropna()

# Clean and convert values to float
cleaned_values = []
for val in values:
    # Extract numeric value from string (e.g., '19.0 (1996)' -> 19.0)
    match = re.search(r'(\d+\.?\d*)', val)
    if match:
        cleaned_values.append(float(match.group(1)))

# Calculate the mean
mean_value = sum(cleaned_values) / len(cleaned_values)
print(f"Final Answer: {mean_value:.1f}")