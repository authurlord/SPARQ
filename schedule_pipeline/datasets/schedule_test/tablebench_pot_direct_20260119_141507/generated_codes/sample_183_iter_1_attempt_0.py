import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert string values with space and minus sign (e.g., '- 7.79') to float
def clean_numeric(x):
    if isinstance(x, str):
        # Remove space before negative numbers
        x = x.strip()
        if x.startswith('- '):
            x = x[2:]
        return float(x)
    return x

# Apply cleaning to relevant columns
df['passengers flown'] = df['passengers flown'].apply(clean_numeric)
df['employees (average / year)'] = df['employees (average / year)'].apply(clean_numeric)
df['basic eps (sek)'] = df['basic eps (sek)'].apply(clean_numeric)
df['net profit / loss (sek)'] = df['net profit / loss (sek)'].apply(clean_numeric)

# Compute pairwise correlation with net profit/loss
correlations = {
    'passengers flown': df['passengers flown'].corr(df['net profit / loss (sek)']),
    'employees (average / year)': df['employees (average / year)'].corr(df['net profit / loss (sek)']),
    'basic eps (sek)': df['basic eps (sek)'].corr(df['net profit / loss (sek)'])
}

# Find the factor with the highest absolute correlation
max_corr = max(abs(correlations[key]) for key in correlations)
if max_corr < 0.3:
    print("Final Answer: no clear impact")
else:
    # Return the factor with the highest absolute correlation
    best_factor = max(correlations, key=lambda k: abs(correlations[k]))
    print(f"Final Answer: {best_factor}")