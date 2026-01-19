import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'property taxes' to numeric
df['property taxes'] = pd.to_numeric(df['property taxes'])
# Extract property taxes for 2000 and 2005
property_taxes_2000 = df[df['year'] == '2000']['property taxes'].values[0]
property_taxes_2005 = df[df['year'] == '2005']['property taxes'].values[0]
# Calculate average annual increase
annual_increase = (property_taxes_2005 - property_taxes_2000) / 5
print(f"Final Answer: {annual_increase:.0f}")