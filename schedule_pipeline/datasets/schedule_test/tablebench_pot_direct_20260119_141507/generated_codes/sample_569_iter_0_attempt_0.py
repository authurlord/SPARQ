import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Injuries (US $000)' to numeric, handling missing values
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')
df['CPSC toy safety funding (US$ Millions)'] = pd.to_numeric(df['CPSC toy safety funding (US$ Millions)'], errors='coerce')

# Filter years with non-null CPSC funding
funding_data = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Extract years and funding values
years = funding_data['Year'].astype(int)
funding = funding_data['CPSC toy safety funding (US$ Millions)']

# Create a list of consecutive 3-year windows
windowed_funding = []
for i in range(len(years) - 2):
    window_years = years.iloc[i:i+3]
    window_avg = funding.iloc[i:i+3].mean()
    windowed_funding.append((window_avg, window_years.tolist()))

# Find the window with the highest average funding
best_window = max(windowed_funding, key=lambda x: x[0])

# Get the years of the best window
best_years = best_window[1]

# Sum the injuries for those years
injury_sum = df[df['Year'].isin(best_years)]['Injuries (US $000)'].sum()

print(f"Final Answer: {injury_sum}")