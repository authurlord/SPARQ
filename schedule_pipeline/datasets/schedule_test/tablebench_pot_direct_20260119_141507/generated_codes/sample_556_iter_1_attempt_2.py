import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Total' column by removing commas and convert to numeric
df['Number of Examinees by Level_4'] = df['Number of Examinees by Level_4'].fillna(0)
df['Total'] = df['Total'].str.replace(',', '').astype(float)

# Filter only Moscow entries
moscow_data = df[df['City'] == 'Moscow']

# Extract the 'Total' values for Moscow by year
moscow_totals = moscow_data[['Year', 'Total']].dropna()

# Convert to numeric and ensure proper order
moscow_totals['Total'] = pd.to_numeric(moscow_totals['Total'], errors='coerce')
moscow_totals = moscow_totals.dropna(subset=['Total'])

# Sort by Year
moscow_totals = moscow_totals.sort_values('Year').reset_index(drop=True)

# Calculate percentage increase from previous year
moscow_totals['percentage_increase'] = (
    (moscow_totals['Total'] - moscow_totals['Total'].shift(1)) / 
    moscow_totals['Total'].shift(1)
)

# Find the year with the largest percentage increase
max_increase_row = moscow_totals.loc[moscow_totals['percentage_increase'].idxmax()]
final_year = max_increase_row['Year']

print(f"Final Answer: {final_year}")