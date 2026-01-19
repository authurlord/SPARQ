import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Extract total and year for both
eu_totals = eu_data.set_index('year')['total'].to_dict()
us_totals = us_data.set_index('year')['total'].to_dict()

# Find common years
common_years = set(eu_totals.keys()) & set(us_totals.keys())

# Check which year EU exceeds US
for year in common_years:
    if eu_totals[year] > us_totals[year]:
        print(f"Final Answer: {year}")
        break