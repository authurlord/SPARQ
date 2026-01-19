import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert all values to float
df = df.apply(pd.to_numeric, errors='coerce')

# Extract column names (each row is a category)
categories = df.columns.tolist()

# Initialize list to store average annual growth rate per category
growth_rates = []

for category in categories:
    # Skip the first year (2006) because we need 5 years of data
    if category == '2006':
        continue
    values = df[category].dropna()
    # We have values from 2006 to 2010 → 5 years → 4 intervals
    years = [2006, 2007, 2008, 2009, 2010]
    data = df.loc[:, '2006':'2010'].values.flatten()
    
    # Reconstruct the data by row
    row_data = df[category].values
    if len(row_data) != 5:
        continue
    
    # Convert to float
    row_data = [float(x) for x in row_data]
    
    # Calculate compound annual growth rate (CAGR)
    initial = row_data[0]
    final = row_data[4]
    n = 4  # number of years between 2006 and 2010
    cagr = (final / initial) ** (1/n) - 1
    growth_rates.append((category, cagr))

# Find the category with the highest CAGR
best_category = max(growth_rates, key=lambda x: x[1])
print(f"Final Answer: {best_category[0]}, {best_category[1]:.2%}")