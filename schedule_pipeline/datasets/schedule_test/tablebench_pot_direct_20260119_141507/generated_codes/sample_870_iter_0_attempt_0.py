import pandas as pd

df = pd.read_csv('table.csv')

# Convert the 'Crude birth rate (per 1000)' and 'Crude death rate (per 1000)' columns to numeric
df['Crude birth rate (per 1000)'] = pd.to_numeric(df['Crude birth rate (per 1000)'], errors='coerce')
df['Crude death rate (per 1000)'] = pd.to_numeric(df['Crude death rate (per 1000)'], errors='coerce')

# Calculate the difference
df['difference'] = df['Crude birth rate (per 1000)'] - df['Crude death rate (per 1000)']

# Find the year with the maximum positive difference
max_diff_row = df[df['difference'] > 0].loc[df['difference'].idxmax()]
year_with_max_margin = max_diff_row['Unnamed: 0']

print(f"Final Answer: {year_with_max_margin}")