import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Convert 'total' column to float for comparison
eu_data['total'] = pd.to_numeric(eu_data['total'])
us_data['total'] = pd.to_numeric(us_data['total'])

# Find the year when EU's total exceeds US's total
for index, row in eu_data.iterrows():
    eu_total = row['total']
    us_total = us_data['total'].iloc[0]  # US data is from 2011
    if eu_total > us_total:
        print(f"Final Answer: {row['year']}")
        break