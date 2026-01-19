import pandas as pd

df = pd.read_csv('table.csv')
# Extract property taxes for years 2000 to 2005
property_taxes = df[df['year'].between(2000, 2005)]['property taxes'].values
# Calculate the average annual increase
total_increase = property_taxes[-1] - property_taxes[0]
average_annual_increase = total_increase / 5
print(f"Final Answer: {average_annual_increase:.0f}")