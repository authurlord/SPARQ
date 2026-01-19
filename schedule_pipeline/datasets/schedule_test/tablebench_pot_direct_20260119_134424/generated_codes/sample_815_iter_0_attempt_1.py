import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop the 'Unnamed: 0' column since it contains region names
df = df.drop(columns=['Unnamed: 0'])

# Select only the year columns
year_columns = [col for col in df.columns if col != 'with fi']

# Convert year columns to numeric, extracting numbers from strings like '19.0 (1996)'
def extract_numeric(val):
    if isinstance(val, str):
        # Extract the first number in the string
        num_str = ''.join(filter(str.isdigit, val))
        if num_str:
            return float(num_str)
        else:
            return np.nan
    return val

# Apply the extraction function to all year columns
df[year_columns] = df[year_columns].applymap(extract_numeric)

# Calculate standard deviation for each region across years
std_devs = df[year_columns].std(axis=1)

# Add the standard deviations back to the dataframe
df['std_dev'] = std_devs

# Sort by standard deviation to find most consistent (low) and inconsistent (high) regions
consistent_region = df.loc[df['std_dev'].idxmin()]
inconsistent_region = df.loc[df['std_dev'].idxmax()]

print(f"Final Answer: {consistent_region.name}, {inconsistent_region.name}")