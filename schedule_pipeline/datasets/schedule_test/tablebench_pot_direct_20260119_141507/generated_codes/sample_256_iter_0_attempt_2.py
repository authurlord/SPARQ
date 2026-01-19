import pandas as pd

df = pd.read_csv('table.csv')

# Remove rows that are headers or totals
# The header row is at index 0, and the total row is at the end
# We'll filter out the last few rows (total rows)
data_rows = df.iloc[1:-3]  # Exclude header and total rows

# Convert relevant columns to numeric
data_rows['Primary'] = pd.to_numeric(data_rows['Primary'], errors='coerce')
data_rows['Career-related'] = pd.to_numeric(data_rows['Career-related'], errors='coerce')

# Sort by Primary schools in descending order
top_3_primary = data_rows.nlargest(3, 'Primary')

# Sum Primary schools for top 3 countries
total_primary_top3 = top_3_primary['Primary'].sum()

# Get total Career-related schools from the total row
total_career_related = df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0]

# Calculate difference
difference = total_primary_top3 - total_career_related

print(f"Final Answer: {difference}")