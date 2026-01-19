import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling strings like '- 7.79'
df['passengers flown'] = pd.to_numeric(df['passengers flown'].str.replace(',', ''), errors='coerce')
df['employees (average / year)'] = pd.to_numeric(df['employees (average / year)'], errors='coerce')
df['net profit / loss (sek)'] = pd.to_numeric(df['net profit / loss (sek)'].str.replace('-', '', regex=False), errors='coerce')
df['basic eps (sek)'] = pd.to_numeric(df['basic eps (sek)'].str.replace('-', '', regex=False), errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna()

# Compute correlation with net profit / loss
correlations = {
    'passengers flown': df['passengers flown'].corr(df['net profit / loss (sek)']),
    'employees (average / year)': df['employees (average / year)'].corr(df['net profit / loss (sek)']),
    'basic eps (sek)': df['basic eps (sek)'].corr(df['net profit / loss (sek)'])
}

# Find the factor with the highest absolute correlation
max_corr = max(abs(correlations[k]) for k in correlations)
if max_corr < 0.3:
    final_answer = 'no clear impact'
else:
    # Return the factor with the highest absolute correlation
    final_answer = max(correlations, key=lambda x: abs(correlations[x]))

print(f"Final Answer: {final_answer}")