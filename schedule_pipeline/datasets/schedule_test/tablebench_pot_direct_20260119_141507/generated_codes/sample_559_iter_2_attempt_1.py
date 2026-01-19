import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter rows where year is between 1895 and 1980 inclusive
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]
# Convert 'floors' to numeric, coercing errors to NaN
filtered_df['floors'] = pd.to_numeric(filtered_df['floors'], errors='coerce')
# Calculate the mean of floors, ignoring NaN values
average_floors = filtered_df['floors'].mean()
print(f"Final Answer: {average_floors:.1f}")