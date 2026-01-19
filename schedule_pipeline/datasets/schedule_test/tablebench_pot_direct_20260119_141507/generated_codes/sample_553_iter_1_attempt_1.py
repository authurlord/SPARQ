import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer for proper numerical operations
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Sort by year to ensure chronological order
df = df.sort_values('year').reset_index(drop=True)

# Extract property taxes values as a series
property_taxes = df['property taxes'].astype(float)

# Calculate the differences between consecutive years
annual_increases = property_taxes.diff()

# Compute the average annual increase
average_increase = annual_increases.mean()

print(f"Final Answer: {average_increase:.0f}")