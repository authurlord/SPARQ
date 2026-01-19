import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, coercing errors to NaN
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is NaN
funding_data = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract the years and funding values
years = funding_data['Year'].astype(int)
funding = funding_data['CPSC toy safety funding (US$ Millions)']

# Initialize variables to track best three-year period
max_avg_funding = -1
best_period_injuries = 0

# Check all possible three-year consecutive periods
for i in range(len(funding) - 2):
    avg_funding = funding.iloc[i:i+3].mean()
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        # Get the sum of injuries for this period
        injuries = df.loc[(df['Year'].astype(int).isin(years.iloc[i:i+3])) & 
                           (df['Injuries (US $000)'] != 'no data'), 'Injuries (US $000)']
        injuries = injuries.astype(float).sum()
        best_period_injuries = injuries

print(f"Final Answer: {best_period_injuries}")