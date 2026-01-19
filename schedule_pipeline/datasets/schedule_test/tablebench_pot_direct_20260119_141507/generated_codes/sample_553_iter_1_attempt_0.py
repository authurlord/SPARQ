import pandas as pd

df = pd.read_csv('table.csv')
# Convert year to integer for proper numerical operations
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Extract property taxes for years 2000 to 2005
property_taxes_2000_2005 = df[df['year'].between(2000, 2005)]['property taxes']

# Calculate annual increases (difference between consecutive years)
increases = []
for i in range(1, len(property_taxes_2000_2005)):
    diff = property_taxes_2000_2005.iloc[i] - property_taxes_2000_2005.iloc[i-1]
    increases.append(diff)

# Compute average increase
average_increase = sum(increases) / len(increases) if increases else 0
print(f"Final Answer: {average_increase:.0f}")