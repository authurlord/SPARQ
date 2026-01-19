import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Convert 'total' column to float for comparison
eu_data['total'] = pd.to_numeric(eu_data['total'])
us_data['total'] = pd.to_numeric(us_data['total'])

# Find the year when EU's total exceeds US's total
# Since data is only available for 2010 (EU) and 2011 (US), compare directly
eu_total_2010 = eu_data[eu_data['year'] == '2010']['total'].values[0]
us_total_2011 = us_data[us_data['year'] == '2011']['total'].values[0]

# Check if EU's total in 2010 exceeds US's total in 2011
if eu_total_2010 > us_total_2011:
    print("Final Answer: 2010")
else:
    print("Final Answer: None")