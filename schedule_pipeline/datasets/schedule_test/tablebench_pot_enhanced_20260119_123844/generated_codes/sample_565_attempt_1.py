import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for years 1850 to 1870
years = [1850, 1855, 1860, 1865, 1870]
data = df[df['Year'].isin(years)]

# Convert values to integers (remove commas)
data['value'] = data['Year_1'].str.replace(',', '').astype(int)

# Calculate differences between consecutive years
data = data.sort_values('Year')
data['increase'] = data['value'].diff()

# Find the year with the highest increase
max_increase_row = data.loc[data['increase'].idxmax()]

# Get the value from the year with the highest increase
final_value = max_increase_row['value']

print(f"Final Answer: {final_value}")