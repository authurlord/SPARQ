import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Injuries (US $000)' to numeric, handling 'no data' and estimates
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Drop rows where funding is missing
valid_funding = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Get the years with valid funding
years_with_funding = valid_funding['Year'].tolist()

# We'll consider all possible 3-year consecutive periods from the valid years
max_avg_funding = -1
best_period_injuries = 0

# Loop through all possible 3-year periods
for i in range(len(years_with_funding) - 2):
    period_years = years_with_funding[i:i+3]
    period_data = df[df['Year'].isin(period_years)]
    avg_funding = period_data['CPSC toy safety funding (US$ Millions)'].mean()
    total_injuries = period_data['Injuries (US $000)'].sum()
    
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        best_period_injuries = total_injuries

print(f"Final Answer: {best_period_injuries}")