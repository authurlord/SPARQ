import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert the 'speed' column to float
df['speed'] = df['speed'].str.replace(' , ', '.').str.strip()
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')

# Convert 'year' to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Sort by year to ensure chronological order
df = df.sort_values('year')

# Calculate the year-on-year change in speed
df['speed_change'] = df['speed'].diff()

# Find the year with the maximum speed increase
max_increase_year = df.loc[df['speed_change'].idxmax(), 'year']

print(f"Final Answer: {max_increase_year}")