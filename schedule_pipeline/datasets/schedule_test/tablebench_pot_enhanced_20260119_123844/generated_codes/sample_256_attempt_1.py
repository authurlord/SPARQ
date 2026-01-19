import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Primary' column to numeric, handling potential formatting issues
df['Primary'] = pd.to_numeric(df['Primary'], errors='coerce')

# Sort by 'Primary' in descending order and get top 3 countries
top_3_primary = df.nlargest(3, 'Primary')['Primary'].sum()

# Get total Career-related schools from the 'Total Schools Globally' row
total_career_related = df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0]
total_career_related = int(total_career_related.replace(',', ''))

# Calculate difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")