import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, coercing errors to NaN
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is missing
df_clean = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract years and funding values
years = df_clean['Year'].values
funding = df_clean['CPSC toy safety funding (US$ Millions)'].values

# Initialize variables to track max average and corresponding injury sum
max_avg_funding = -1
total_injuries = 0

# Check all possible 3-year periods
for i in range(len(funding) - 2):
    period_funding = funding[i:i+3]
    avg_funding = sum(period_funding) / 3
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        # Get the corresponding injury values for this period
        injuries = [df_clean.loc[df_clean['Year'] == year, 'Injuries (US $000)'].values[0] for year in years[i:i+3]]
        total_injuries = sum(int(inj.replace(' (estimate)', '')) if isinstance(inj, str) and 'estimate' in inj else int(inj) for inj in injuries)

print(f"Final Answer: {total_injuries}")