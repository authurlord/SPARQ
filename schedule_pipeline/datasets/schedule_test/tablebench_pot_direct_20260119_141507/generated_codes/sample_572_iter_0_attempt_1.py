import pandas as pd

df = pd.read_csv('table.csv')
# Convert year column to integer for proper ordering
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Extract typhus cases and compute year-over-year differences
typhus_cases = df['typhus'].astype(float)
differences = typhus_cases.diff()

# Find the year with the maximum decrease (most negative)
decrease_index = differences.idxmax()  # This will give us the index of the max negative drop
# Since the index is not directly available, we need to find the row where the difference is minimum (most negative)
min_decrease_idx = differences.argmin()
year_with_max_decrease = df.iloc[min_decrease_idx]['year']

print(f"Final Answer: {year_with_max_decrease}")