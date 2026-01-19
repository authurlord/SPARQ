import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'speed' column: remove spaces and commas, then convert to float
def clean_speed(speed):
    # Replace comma and space with empty string, then convert to float
    return float(speed.replace(',', '').replace(' ', ''))

df['speed'] = df['speed'].apply(clean_speed)

# Convert 'year' to integer type
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Drop rows with invalid year or speed
df = df.dropna(subset=['year', 'speed'])

# Group by year and calculate average speed per year
avg_speed_by_year = df.groupby('year')['speed'].mean()

# Compute the difference in average speed from previous year
avg_speed_by_year = avg_speed_by_year.reset_index()
avg_speed_by_year['diff_from_prev'] = avg_speed_by_year['speed'].diff()

# Find the year with the maximum increase (positive difference)
max_increase_year = avg_speed_by_year.loc[avg_speed_by_year['diff_from_prev'].idxmax(), 'year']

print(f"Final Answer: {max_increase_year}")