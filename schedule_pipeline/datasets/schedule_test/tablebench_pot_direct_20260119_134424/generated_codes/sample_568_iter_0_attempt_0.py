import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Compare total energy production by year
eu_total = eu_data['total'].values[0]
us_total = us_data['total'].values[0]

# Check if EU's total exceeds US's total
if eu_total > us_total:
    print(f"Final Answer: {eu_data['year'].values[0]}")
else:
    print("Final Answer: None")