import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert year and total s ton to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['total s ton'] = pd.to_numeric(df['total s ton'], errors='coerce')

# Sort by year in ascending order
df = df.sort_values(by='year').reset_index(drop=True)

# Calculate year-over-year increase in total s ton
df['increase'] = df['total s ton'].diff()

# Find the year with the highest increase (excluding the first row, which has no previous year)
max_increase_row = df[df['increase'] == df['increase'].max()]
max_increase_year = max_increase_row.iloc[0]['year']

print(f"Final Answer: {max_increase_year}")