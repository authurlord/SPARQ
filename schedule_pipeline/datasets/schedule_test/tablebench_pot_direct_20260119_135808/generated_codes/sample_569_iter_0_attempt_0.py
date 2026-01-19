import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'CPSC toy safety funding (US$ Millions)' to numeric, handling 'no data' and '13.6' etc.
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Filter rows with valid funding data
valid_funding = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Calculate average funding for each consecutive 3-year period
max_avg_funding = 0
best_period_injuries = 0

for i in range(len(valid_funding) - 2):
    subset = valid_funding.iloc[i:i+3]
    avg_funding = subset['CPSC toy safety funding (US$ Millions)'].mean()
    if avg_funding > max_avg_funding:
        max_avg_funding = avg_funding
        # Get the injuries for this period
        injuries = subset['Injuries (US $000)'].replace('no data', None).astype(float).sum()
        best_period_injuries = injuries

print(f"Final Answer: {best_period_injuries}")