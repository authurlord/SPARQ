import pandas as pd

df = pd.read_csv('table.csv')

# Filter out header and total rows
data_rows = df[df['Country'] != '-']
data_rows = data_rows[data_rows['Country'] != 'Total Schools Globally']
data_rows = data_rows[data_rows['Country'] != 'Countries & Territories']

# Convert relevant columns to numeric (remove commas)
data_rows['Primary'] = data_rows['Primary'].str.replace(',', '').astype(int)
data_rows['Career-related'] = data_rows['Career-related'].str.replace(',', '').astype(int)

# Sort by Primary schools in descending order and take top 3
top_3_primary = data_rows.nlargest(3, 'Primary')[['Primary']].sum()

# Total Career-related schools across all countries
total_career_related = data_rows['Career-related'].sum()

# Compute difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")