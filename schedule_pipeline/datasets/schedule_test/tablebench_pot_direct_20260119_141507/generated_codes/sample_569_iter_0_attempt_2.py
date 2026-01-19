import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Year' to integer for proper sorting
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Filter rows where CPSC toy safety funding is not null
df_filtered = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Sort by Year
df_filtered = df_filtered.sort_values('Year').reset_index(drop=True)

# Extract years and funding values
years = df_filtered['Year'].values
funding = df_filtered['CPSC toy safety funding (US$ Millions)'].values

# Create list of possible 3-year consecutive windows
window_funding_avg = []
for i in range(len(years) - 2):
    window_years = years[i:i+3]
    window_funding = funding[i:i+3]
    avg_funding = np.mean(window_funding)
    window_funding_avg.append((avg_funding, window_years))

# Find the window with the highest average funding
max_avg = max(window_funding_avg, key=lambda x: x[0])[0]
best_window_years = [int(y) for y in max(window_funding_avg, key=lambda x: x[0])[1]]

# Now get the injuries for those years from original data
injury_sum = 0
for year in best_window_years:
    row = df[df['Year'] == year]
    if not row.empty:
        injuries = row['Injuries (US $000)'].values[0]
        injury_sum += injuries

print(f"Final Answer: {injury_sum}")