import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'property taxes' to integers
df['property taxes'] = df['property taxes'].astype(int)
# Extract values for 2000 to 2005
property_taxes = df['property taxes'].values
# Calculate annual increases
annual_increases = [property_taxes[i] - property_taxes[i-1] for i in range(1, len(property_taxes))]
# Calculate average annual increase
avg_increase = sum(annual_increases) / len(annual_increases)
print(f"Final Answer: {avg_increase:.0f}")