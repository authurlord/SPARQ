import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract the '2001 general' column
column_name = '2001 general'
values = df[column_name].dropna()

# Clean each value: remove text in parentheses and extract numbers
cleaned_values = []
for val in values:
    # Remove text in parentheses and extract the number
    match = re.search(r'(\d+\.?\d*)', val)
    if match:
        cleaned_values.append(float(match.group(1)))

# Calculate the mean of cleaned values
mean_2001 = sum(cleaned_values) / len(cleaned_values)

print(f"Final Answer: {mean_2001:.1f}")