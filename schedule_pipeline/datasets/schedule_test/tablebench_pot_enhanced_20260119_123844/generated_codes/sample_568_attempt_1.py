import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to float for comparison
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Filter data for EU and US
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Extract total energy production for each country by year
eu_total = eu_data.set_index('year')['total'].to_dict()
us_total = us_data.set_index('year')['total'].to_dict()

# Find the year where EU's total exceeds US's total
for year in eu_total:
    if year in us_total and eu_total[year] > us_total[year]:
        print(f"Final Answer: {year}")
        break