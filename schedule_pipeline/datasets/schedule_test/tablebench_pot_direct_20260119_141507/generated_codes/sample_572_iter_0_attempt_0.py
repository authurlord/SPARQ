import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove or handle invalid entries
# The row with '191822' is invalid; we'll drop it or skip it
df_clean = df[df['year'].str.contains(r'\d{4}')].copy()  # Keep only rows with valid 4-digit years

# Convert 'year' to integer and 'typhus' to integer
df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
df_clean['typhus'] = pd.to_numeric(df_clean['typhus'], errors='coerce')

# Sort by year
df_clean = df_clean.sort_values('year').dropna()

# Calculate the year-on-year change in typhus cases
df_clean['typhus_diff'] = df_clean['typhus'].diff()

# Find the year with the maximum decrease (most negative)
max_decrease_year = df_clean.loc[df_clean['typhus_diff'].idxmin(), 'year']

print(f"Final Answer: {max_decrease_year}")