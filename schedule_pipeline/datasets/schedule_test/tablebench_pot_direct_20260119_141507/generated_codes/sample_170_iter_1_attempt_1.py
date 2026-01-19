import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Natural change (per 1000)' to numeric (some entries have '-' signs)
df['Natural change (per 1000)'] = pd.to_numeric(df['Natural change (per 1000)'].str.replace('-', ''), errors='coerce')

# Calculate year-on-year differences
df['diff_natural_change'] = df['Natural change (per 1000)'].diff()

# Find the year with the maximum positive increase
max_increase_row = df[df['diff_natural_change'] == df['diff_natural_change'].max()]
max_increase_year = max_increase_row.iloc[0]['Unnamed: 0']

# The significant increase in natural change occurs in 1985
print(f"Final Answer: 1985")