import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, coercing errors to NaN
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is NaN
valid_funding = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Sort by year to ensure chronological order
valid_funding = valid_funding.sort_values('Year').reset_index(drop=True)

# Calculate rolling average over 3 years
valid_funding['rolling_avg_funding'] = valid_funding['CPSC toy safety funding (US$ Millions)'].rolling(window=3).mean()

# Find the row with the maximum rolling average
max_avg_row = valid_funding.loc[valid_funding['rolling_avg_funding'].idxmax()]

# Get the three years corresponding to this peak average
start_year = max_avg_row['Year']
# Find the three consecutive years around this peak
start_idx = valid_funding[valid_funding['Year'] == start_year].index[0]
three_years = valid_funding.iloc[start_idx:start_idx + 3]

# Extract injury values for these three years
injuries = three_years['Injuries (US $000)']

# Convert injuries to numeric, handling 'no data' and '202 (estimate)'
injuries = pd.to_numeric(injuries, errors='coerce')

# Sum the injuries (in thousands)
total_injuries = injuries.sum()

print(f"Final Answer: {total_injuries:.0f}")