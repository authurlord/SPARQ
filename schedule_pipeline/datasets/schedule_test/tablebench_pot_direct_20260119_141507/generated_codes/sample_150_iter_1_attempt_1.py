import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'population (000)' to numeric and ensure it's in the correct format
df['population (000)'] = pd.to_numeric(df['population (000)'], errors='coerce')

# Sort by year (January)
df = df.sort_values('year (january)').reset_index(drop=True)

# Calculate year-over-year population growth rate (as percentage)
df['growth_rate'] = df['population (000)'].pct_change() * 100

# Drop the first row (no prior year)
df = df.dropna(subset=['growth_rate'])

# Extract urbanization percentage and growth rate
urbanization = df['urban , %']
growth_rate = df['growth_rate']

# Compute correlation between urbanization and population growth rate
correlation = urbanization.corr(growth_rate)

print(f"Final Answer: {correlation:.2f}")