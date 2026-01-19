import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter rows for years between 2004 and 2013 inclusive
filtered_df = df[(df['Year'].astype(str).str.startswith('200')) & (df['Year'].between(2004, 2013))]

# Convert 'No. of performances' to numeric, coercing errors to NaN
performances_numeric = pd.to_numeric(filtered_df['No. of performances'], errors='coerce')

# Calculate the mean of valid performances
mean_performances = performances_numeric.mean()

print(f"Final Answer: {mean_performances:.1f}")