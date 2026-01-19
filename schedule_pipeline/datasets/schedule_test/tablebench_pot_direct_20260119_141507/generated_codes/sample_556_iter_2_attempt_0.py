import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where City is 'Moscow'
moscow_data = df[df['City'] == 'Moscow']

# Clean the 'Total' column: remove NaN or invalid entries like '-' or 'Data missing'
moscow_data['Total'] = pd.to_numeric(moscow_data['Total'], errors='coerce')
moscow_data = moscow_data.dropna(subset=['Total'])

# Convert to numeric and sort by Year
moscow_data['Year'] = pd.to_numeric(moscow_data['Year'], errors='coerce')
moscow_data = moscow_data.sort_values('Year').reset_index(drop=True)

# Calculate year-over-year percentage increase
moscow_data['previous_total'] = moscow_data['Total'].shift(1)
moscow_data['percentage_increase'] = ((moscow_data['Total'] - moscow_data['previous_total']) / moscow_data['previous_total']) * 100

# Find the year with the largest percentage increase
max_increase_row = moscow_data.loc[moscow_data['percentage_increase'].idxmax()]
final_year = max_increase_row['Year']

print(f"Final Answer: {final_year}")