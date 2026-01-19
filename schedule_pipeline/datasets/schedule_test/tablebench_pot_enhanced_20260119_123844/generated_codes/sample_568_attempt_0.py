import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Convert 'total' column to float for comparison
eu_data['total'] = pd.to_numeric(eu_data['total'])
us_data['total'] = pd.to_numeric(us_data['total'])

# Find the year where EU's total exceeds US's total
for _, row in eu_data.iterrows():
    eu_total = row['total']
    us_row = us_data[us_data['year'] == row['year']]
    if not us_row.empty:
        us_total = us_row['total'].values[0]
        if eu_total > us_total:
            print(f"Final Answer: {row['year']}")
            break