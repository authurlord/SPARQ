import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where City is 'Moscow'
moscow_data = df[df['City'] == 'Moscow']

# Convert 'Year' to integer and 'Number of Examinees by Level_4' to numeric (Total column)
moscow_data['Year'] = pd.to_numeric(moscow_data['Year'], errors='coerce')
moscow_data['Total'] = pd.to_numeric(moscow_data['Total'], errors='coerce')

# Remove rows with missing Total values
moscow_data = moscow_data.dropna(subset=['Total'])

# Sort by Year
moscow_data = moscow_data.sort_values('Year')

# Calculate year-over-year percentage increase
moscow_data['Previous_Total'] = moscow_data['Total'].shift(1)
moscow_data['Percentage_Increase'] = ((moscow_data['Total'] - moscow_data['Previous_Total']) / moscow_data['Previous_Total']) * 100

# Find the year with the largest percentage increase
max_increase_row = moscow_data.loc[moscow_data['Percentage_Increase'].idxmax()]
final_year = max_increase_row['Year']

print(f"Final Answer: {final_year}")