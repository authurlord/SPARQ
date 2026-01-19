import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Primary' and 'Career-related' columns to numeric, handling potential formatting issues
df['Primary'] = pd.to_numeric(df['Primary'], errors='coerce')
df['Career-related'] = pd.to_numeric(df['Career-related'], errors='coerce')

# Drop rows with missing country names (like '---' or 'Total Schools Globally')
df_clean = df.dropna(subset=['Country'])

# Sort by 'Primary' in descending order and get top 3 countries
top_3_primary = df_clean.nlargest(3, 'Primary')['Primary'].sum()

# Get total Career-related schools from the 'Total Schools Globally' row
total_career_related = df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0]

# Calculate the difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")