import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Extract the year and total values
eu_total = eu_data['total'].values[0]
us_total = us_data['total'].values[0]

# Compare totals
if eu_total > us_total:
    final_year = eu_data['year'].values[0]
else:
    final_year = us_data['year'].values[0]

print(f"Final Answer: {final_year}")