import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, handling 'no data' and strings
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is missing
valid_data = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract relevant columns
years = valid_data['Year'].astype(int)
funding = valid_data['CPSC toy safety funding (US$ Millions)']
injuries = valid_data['Injuries (US $000)']

# Convert injuries to numeric, handle 'no data' and '202 (estimate)'
injuries = injuries.replace('no data', None)
injuries = pd.to_numeric(injuries, errors='coerce')

# Combine into a DataFrame for analysis
valid_data = pd.DataFrame({
    'Year': years,
    'Funding': funding,
    'Injuries': injuries
})

# Sort by Year
valid_data = valid_data.sort_values('Year')

# Find all consecutive 3-year periods
max_avg_funding = -1
best_period_injuries = 0

for i in range(len(valid_data) - 2):
    subset = valid_data.iloc[i:i+3]
    avg_funding = subset['Funding'].mean()
    total_injuries = subset['Injuries'].sum()
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        best_period_injuries = total_injuries

print(f"Final Answer: {best_period_injuries}")