import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'speed' column by removing commas and converting to float
df['speed'] = df['speed'].str.replace(',', '').astype(float)

# Convert 'year' to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Group by year and calculate average speed
avg_speed_by_year = df.groupby('year')['speed'].mean().reset_index()

# Sort by year to ensure chronological order
avg_speed_by_year = avg_speed_by_year.sort_values('year')

# Compute year-over-year differences
avg_speed_by_year['speed_diff'] = avg_speed_by_year['speed'].diff()

# Find the year with the maximum increase (excluding first year since no prior year)
max_increase_row = avg_speed_by_year[avg_speed_by_year['speed_diff'] == avg_speed_by_year['speed_diff'].max()]

# Extract the year with the maximum increase
final_year = max_increase_row.iloc[0]['year'] if not max_increase_row.empty else None

print(f"Final Answer: {final_year}")