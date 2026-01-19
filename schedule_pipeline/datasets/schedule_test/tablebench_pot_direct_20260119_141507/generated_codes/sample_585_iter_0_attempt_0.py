import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer and 'number of tropical storms' to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'], errors='coerce')

# Sort by year
df = df.sort_values('year').reset_index(drop=True)

# Calculate year-over-year difference in tropical storms
df['diff'] = df['number of tropical storms'].diff()

# Find the year with the maximum increase (excluding the first row which has no prior year)
max_increase_row = df[df['diff'] == df['diff'].max()]
year_with_max_increase = max_increase_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_increase}")