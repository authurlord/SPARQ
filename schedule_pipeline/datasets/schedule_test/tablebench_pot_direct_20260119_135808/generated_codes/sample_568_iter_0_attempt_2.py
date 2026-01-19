import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to float for comparison
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Extract total energy production for each country
eu_total = eu_data['total'].values[0]
us_total = us_data['total'].values[0]

# Compare and determine the year
if eu_total > us_total:
    year = eu_data['year'].values[0]
else:
    year = None

print(f"Final Answer: {year}")