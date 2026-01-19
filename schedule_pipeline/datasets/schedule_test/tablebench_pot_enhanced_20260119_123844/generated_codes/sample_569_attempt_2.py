import pandas as pd

df = pd.read_csv('table.csv')

# Convert CPSC toy safety funding to numeric, coercing errors to NaN
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is missing
valid_funding = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract the years and funding values
years = valid_funding['Year'].values
funding = valid_funding['CPSC toy safety funding (US$ Millions)'].values

# Find all consecutive 3-year periods and compute average funding
max_avg_funding = -1
best_period_injuries = 0

for i in range(len(funding) - 2):
    period_funding = funding[i:i+3]
    avg_funding = sum(period_funding) / 3
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        # Get corresponding injuries for this period
        period_years = years[i:i+3]
        injuries_sum = 0
        for year in period_years:
            row = df[df['Year'] == year]
            injury_val = row['Injuries (US $000)'].iloc[0]
            if injury_val == 'no data':
                continue
            # Remove 'estimate' and convert to float
            injury_clean = injury_val.replace(' (estimate)', '')
            injuries_sum += float(injury_clean)
        best_period_injuries = injuries_sum

print(f"Final Answer: {best_period_injuries}")