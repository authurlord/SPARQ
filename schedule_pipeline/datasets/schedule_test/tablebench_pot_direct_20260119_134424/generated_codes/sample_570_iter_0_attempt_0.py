import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'speed' column to float
df['speed'] = df['speed'].str.replace(',', '.').astype(float)
# Convert 'year' to int
df['year'] = df['year'].astype(int)
# Sort by year
df = df.sort_values('year')
# Calculate the difference in speed between consecutive years
df['speed_diff'] = df['speed'].diff()
# Find the year with the maximum speed increase
max_increase_year = df.loc[df['speed_diff'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")