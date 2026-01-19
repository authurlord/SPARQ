import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for EU and US
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Convert 'total' column to float for comparison
eu_data['total'] = pd.to_numeric(eu_data['total'])
us_data['total'] = pd.to_numeric(us_data['total'])

# Find the year when EU's total exceeds US's total
# Since EU data is from 2010 and US from 2011, no overlapping year exists
# But check if EU's 2010 value > US's 2011 value
eu_2010_total = eu_data[eu_data['year'] == '2010']['total'].values[0]
us_2011_total = us_data[us_data['year'] == '2011']['total'].values[0]

if eu_2010_total > us_2011_total:
    print(f"Final Answer: 2010")
else:
    print(f"Final Answer: None")