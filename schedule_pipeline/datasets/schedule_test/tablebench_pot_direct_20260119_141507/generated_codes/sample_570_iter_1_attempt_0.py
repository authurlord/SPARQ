import pandas as pd
import numpy as np

# Load the dataframe
df = pd.read_csv('table.csv')

# Clean the 'speed' column: remove spaces and commas, then convert to float
def clean_speed(speed_str):
    # Replace comma and space with decimal point
    cleaned = speed_str.replace(',', '.').replace(' ', '')
    return float(cleaned)

# Apply cleaning to the 'speed' column
df['speed'] = df['speed'].apply(clean_speed)

# Convert 'year' to integer type
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Remove any rows with invalid year or speed
df = df.dropna(subset=['year', 'speed'])

# Group by year and calculate average speed per year
avg_speed_per_year = df.groupby('year')['speed'].mean()

# Compute year-over-year increase
diffs = avg_speed_per_year.diff()

# Find the year with the maximum increase (positive)
max_increase_year = diffs.idxmax()

# If no increase (all zero or negative), return the first valid year
if pd.isna(max_increase_year):
    max_increase_year = df['year'].iloc[0]

print(f"Final Answer: {max_increase_year}")