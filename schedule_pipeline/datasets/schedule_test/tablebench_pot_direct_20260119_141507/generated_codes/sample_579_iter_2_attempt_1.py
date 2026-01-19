import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'year' column: split strings like "2010 , 2012" into individual years
def extract_years(year_str):
    if isinstance(year_str, str):
        # Split by comma and space, then convert to integers
        years = [int(y.strip()) for y in year_str.split(', ') if y.strip().isdigit()]
        return years
    return []

# Apply the function to create a list of years for each row
df['years'] = df['year'].apply(extract_years)

# Create a boolean mask for rows where any year is in 2000-2007
mask = df['years'].apply(lambda x: any(y >= 2000 and y <= 2007 for y in x))

# Filter data for those years
filtered_df = df[mask]

# Compute average quantity
avg_quantity = filtered_df['quantity'].mean()

print(f"Final Answer: {avg_quantity:.1f}")