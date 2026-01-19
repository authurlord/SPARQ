import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Remove the first row (header) if it's duplicated or incorrectly formatted
# The actual header is already in columns, so we drop the first row which is metadata
# Filter only Moscow entries
df_moscow = df[df['City'] == 'Moscow'].copy()

# Drop rows where 'Total' is missing or invalid (e.g., '-')
df_moscow = df_moscow.dropna(subset=['Number of Examinees by Level_4'])  # Ensure Total is present
df_moscow = df_moscow[df_moscow['Total'].notna() & (df_moscow['Total'] != '-')]

# Convert Year to integer
df_moscow['Year'] = pd.to_numeric(df_moscow['Year'], errors='coerce')
df_moscow = df_moscow.dropna(subset=['Year'])

# Sort by Year
df_moscow = df_moscow.sort_values('Year').reset_index(drop=True)

# Extract Total values
totals = df_moscow['Total'].astype(int)

# Compute year-over-year percentage increase
yearly_increases = []
for i in range(1, len(totals)):
    prev = totals[i-1]
    curr = totals[i]
    if prev > 0:
        increase = ((curr - prev) / prev) * 100
        yearly_increases.append(increase)

# Find the maximum increase and corresponding year
if len(yearly_increases) > 0:
    max_increase_idx = np.argmax(yearly_increases)
    max_increase_year = df_moscow['Year'].iloc[max_increase_idx + 1]
    print(f"Final Answer: {max_increase_year}")
else:
    print("Final Answer: No valid data for percentage increase")