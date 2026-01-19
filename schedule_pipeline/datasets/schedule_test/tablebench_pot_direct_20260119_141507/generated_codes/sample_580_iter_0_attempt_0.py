import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert all relevant columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Calculate year-over-year differences for 'bötzow'
df['bötzow_diff'] = df['bötzow'].diff()

# Find the year with the maximum increase (ignore NaN for first row)
max_increase_row = df[df['bötzow_diff'] == df['bötzow_diff'].max()]
year_with_max_increase = max_increase_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_increase}")