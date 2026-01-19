import pandas as pd
import re

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'speed' column: remove commas and convert to float
def clean_speed(speed):
    # Remove commas and convert to float
    return float(re.sub(r',', '', str(speed)))

df['speed'] = df['speed'].apply(clean_speed)

# Convert 'year' to integer type
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Remove rows with invalid year or speed
df = df.dropna(subset=['year', 'speed'])

# Group by year and calculate average speed per year
avg_speed_by_year = df.groupby('year')['speed'].mean()

# Calculate year-over-year differences
diffs = avg_speed_by_year.diff().dropna()

# Find the year with the maximum increase
max_increase_year = diffs.idxmax()

print(f"Final Answer: {max_increase_year}")