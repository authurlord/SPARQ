import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Primary' column to numeric, handling potential formatting issues (e.g., commas)
df['Primary'] = pd.to_numeric(df['Primary'].str.replace(',', ''), errors='coerce')

# Drop rows with missing data (like the header row with '-')
df_clean = df.dropna(subset=['Country'])

# Sort by Primary schools in descending order and get top 3
top_3_primary = df_clean.nlargest(3, 'Primary')['Primary'].sum()

# Get total Career-related schools from the "Total Schools Globally" row
total_career_related = df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0]
total_career_related = int(total_career_related.replace(',', ''))

# Calculate difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")