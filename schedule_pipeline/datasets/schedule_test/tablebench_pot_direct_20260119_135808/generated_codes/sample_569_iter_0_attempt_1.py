import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, coercing errors to NaN
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is NaN
valid_funding = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract years and funding values
years = valid_funding['Year'].tolist()
funding = valid_funding['CPSC toy safety funding (US$ Millions)'].tolist()

# Calculate average funding for every 3-year consecutive period
max_avg_funding = -1
best_period_injuries = 0

for i in range(len(funding) - 2):
    avg_funding = sum(funding[i:i+3]) / 3
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        # Get injuries for this 3-year period
        injuries = sum(pd.to_numeric(df.loc[df['Year'] == year, 'Injuries (US $000)'], errors='coerce') for year in years[i:i+3])
        best_period_injuries = injuries

print(f"Final Answer: {best_period_injuries}")