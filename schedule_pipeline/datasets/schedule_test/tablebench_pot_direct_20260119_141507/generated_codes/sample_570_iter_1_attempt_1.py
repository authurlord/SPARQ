import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'speed' column: remove spaces and commas, then convert to float
df['speed'] = df['speed'].str.replace(',', '').str.replace(' ', '').astype(float)

# Convert 'year' to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Compute average speed per year
avg_speed_by_year = df.groupby('year')['speed'].mean()

# Calculate the difference in average speed from previous year
avg_speed_by_year = avg_speed_by_year.reset_index()
avg_speed_by_year['diff'] = avg_speed_by_year['speed'].diff()

# Find the year with the maximum increase (positive difference)
max_increase_year = avg_speed_by_year.loc[avg_speed_by_year['diff'].idxmax(), 'year']

print(f"Final Answer: {max_increase_year}")