import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Total' column by removing commas and converting to float
df['Total'] = df['Total'].astype(str).str.replace(',', '').astype(float)

# Filter rows where City is 'Moscow'
moscow_data = df[df['City'] == 'Moscow']

# Convert Year to numeric and sort by Year
moscow_data['Year'] = pd.to_numeric(moscow_data['Year'], errors='coerce')
moscow_data = moscow_data.dropna(subset=['Year', 'Total'])

# Sort by Year
moscow_data = moscow_data.sort_values('Year')

# Calculate percentage increase from previous year
moscow_data['percentage_increase'] = moscow_data['Total'].pct_change() * 100

# Find the year with the largest positive percentage increase
max_increase_row = moscow_data.loc[moscow_data['percentage_increase'].idxmax()]
final_year = max_increase_row['Year']

print(f"Final Answer: {final_year}")