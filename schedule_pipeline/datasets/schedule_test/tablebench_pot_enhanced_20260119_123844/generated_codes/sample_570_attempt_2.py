import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'speed' column: remove commas and convert to float
df['speed'] = df['speed'].str.replace(',', '').astype(float)

# Convert 'year' to integer
df['year'] = df['year'].astype(int)

# Sort by year to ensure chronological order
df = df.sort_values('year')

# Calculate year-on-year change in speed
df['speed_change'] = df['speed'].diff()

# Find the year with the maximum positive increase
max_increase_year = df.loc[df['speed_change'].idxmax(), 'year']

print(f"Final Answer: {max_increase_year}")