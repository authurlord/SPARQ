import pandas as pd

df = pd.read_csv('table.csv')
# Convert year column to integer and sort by year
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.sort_values('year').dropna()

# Extract typhus cases
typhus_cases = df['typhus'].astype(float)

# Compute year-over-year differences
differences = typhus_cases[1:] - typhus_cases[:-1]

# Find the year with the largest decrease (most negative difference)
decrease_idx = differences.argmin()
target_year = df['year'].iloc[decrease_idx + 1]  # The year after the previous one

print(f"Final Answer: {target_year}")