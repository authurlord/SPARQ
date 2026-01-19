import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Total' column by removing commas and converting to numeric
df['Number of Examinees by Level_4'] = df['Number of Examinees by Level_4'].str.replace(',', '').astype(float)
df['Total'] = df['Total'].str.replace(',', '').astype(float)

# Filter only Moscow entries
moscow_data = df[df['City'] == 'Moscow']

# Extract years and total examinees
moscow_years = moscow_data['Year'].dropna().astype(int)
moscow_totals = moscow_data['Total'].dropna()

# Convert to list for easier processing
totals = moscow_totals.tolist()
years = moscow_years.tolist()

# Remove missing values and ensure we have at least 2 years
if len(totals) < 2:
    print("Not enough data to compute percentage increase.")
else:
    # Calculate percentage increase from one year to the next
    increases = []
    for i in range(1, len(totals)):
        prev = totals[i-1]
        curr = totals[i]
        if prev == 0:
            continue
        increase = ((curr - prev) / prev) * 100
        increases.append(increase)
    
    # Find the index of maximum increase
    max_increase_idx = increases.index(max(increases)) if increases else None
    
    # Get the corresponding year (the year after the previous year)
    if max_increase_idx is not None:
        result_year = years[max_increase_idx + 1]
        print(f"Final Answer: {result_year}")
    else:
        print("Final Answer: No valid increase found")