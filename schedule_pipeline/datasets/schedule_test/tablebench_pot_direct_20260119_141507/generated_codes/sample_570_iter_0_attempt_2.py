import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'speed' column by removing commas and converting to float
df['speed'] = df['speed'].str.replace(',', '').astype(float)

# Calculate average speed per year
avg_speed_by_year = df.groupby('year')['speed'].mean()

# Compute year-over-year differences
speed_diff = avg_speed_by_year.diff()

# Find the year with the maximum increase (excluding the first year which has no prior)
max_increase_year = speed_diff.idxmax()

print(f"Final Answer: {max_increase_year}")