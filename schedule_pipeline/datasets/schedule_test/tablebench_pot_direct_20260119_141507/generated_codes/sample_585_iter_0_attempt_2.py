import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'year' to integer for proper sorting and calculation
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Sort by year
df = df.sort_values('year').reset_index(drop=True)

# Extract the number of tropical storms
storms = df['number of tropical storms'].astype(int)

# Calculate year-over-year differences
differences = storms[1:] - storms[:-1]

# Find the index of the maximum increase
max_increase_idx = differences.argmax()

# Get the corresponding year (which is the year after the first year in the difference)
target_year = df['year'].iloc[max_increase_idx + 1]

print(f"Final Answer: {target_year}")