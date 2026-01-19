import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Primary' column to numeric (remove commas and handle strings)
df['Primary'] = df['Primary'].str.replace(',', '').astype(int)

# Get top 3 countries by Primary schools
top_3_primary = df.nlargest(3, 'Primary')['Primary'].sum()

# Get total Career-related schools
total_career_related = df[df['Country'] == 'Total Schools Globally']['Career-related'].values[0]
total_career_related = int(total_career_related.replace(',', ''))

# Calculate difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")