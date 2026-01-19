import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'speed' column to float
df['speed'] = df['speed'].str.replace(',', '.').astype(float)
# Sort by year to ensure chronological order
df = df.sort_values('year').reset_index(drop=True)
# Calculate year-on-year speed difference
df['speed_diff'] = df['speed'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['speed_diff'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")