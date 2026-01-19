import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Primary' column to numeric (remove commas and handle strings)
df['Primary'] = df['Primary'].str.replace(',', '').astype(int)

# Sort by 'Primary' in descending order and get top 3 countries
top_3_primary = df.nlargest(3, 'Primary')['Primary'].sum()

# Get total Career-related schools (from the row with 'Total Schools Globally')
total_career_related = df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0]
total_career_related = int(total_career_related.replace(',', ''))

# Calculate difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")