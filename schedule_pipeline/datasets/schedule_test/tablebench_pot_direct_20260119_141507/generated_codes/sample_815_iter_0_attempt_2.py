import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove text in parentheses and convert to float
def clean_value(x):
    if isinstance(x, str):
        # Remove content in parentheses
        x = x.split('(')[0].strip()
        # Handle cases like '19.0 (1996)' -> '19.0'
        if x.endswith(')'):
            x = x[:-1]
        try:
            return float(x)
        except ValueError:
            return np.nan
    return x

# Apply cleaning to all columns except the first (Unnamed: 0)
columns_to_clean = df.columns[1:]  # Exclude 'Unnamed: 0'
cleaned_data = df[columns_to_clean].applymap(clean_value)

# Compute standard deviation for each region (row)
std_dev_by_region = cleaned_data.std(axis=1)

# Get region names (from the first column)
regions = df.iloc[:, 0]

# Create a DataFrame to pair region with standard deviation
std_df = pd.DataFrame({'region': regions, 'std_dev': std_dev_by_region})

# Find the region with the lowest and highest standard deviation
most_consistent = std_df.loc[std_df['std_dev'].idxmin()]
most_inconsistent = std_df.loc[std_df['std_dev'].idxmax()]

print(f"Final Answer: {most_consistent['region']}, {most_inconsistent['region']}")