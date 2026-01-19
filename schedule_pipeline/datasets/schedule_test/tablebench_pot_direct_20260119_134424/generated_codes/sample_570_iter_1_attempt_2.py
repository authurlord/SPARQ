import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'speed' column: remove spaces and replace comma with dot
df['speed'] = df['speed'].str.replace(' ', '').str.replace(',', '.')

# Convert to float
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')

# Convert 'year' to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Sort by year
df = df.sort_values('year')

# Calculate year-on-year speed difference
df['speed_diff'] = df['speed'].diff()

# Find the year with the maximum increase in speed
max_increase_year = df.loc[df['speed_diff'].idxmax(), 'year']

print(f"Final Answer: {int(max_increase_year)}")