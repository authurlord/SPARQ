import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, coercing errors to NaN
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is NaN
valid_funding = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract years and funding values
years = valid_funding['Year'].astype(int)
funding = valid_funding['CPSC toy safety funding (US$ Millions)']

# Slide a 3-year window and compute average funding
max_avg_funding = -1
best_period_years = []

for i in range(len(years) - 2):
    window_years = years[i:i+3]
    window_funding = funding[i:i+3]
    avg_funding = window_funding.mean()
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        best_period_years = window_years.tolist()

# Now get injuries for those three years
injury_data = df[['Year', 'Injuries (US $000)']]
injury_data['Year'] = injury_data['Year'].astype(int)

# Filter injuries for the best period
injuries = injury_data[injury_data['Year'].isin(best_period_years)]
total_injuries = injuries['Injuries (US $000)'].sum()

print(f"Final Answer: {total_injuries}")