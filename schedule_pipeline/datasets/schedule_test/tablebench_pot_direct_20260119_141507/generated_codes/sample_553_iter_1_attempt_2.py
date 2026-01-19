import pandas as pd

df = pd.read_csv('table.csv')
# Convert year to integer for proper numerical operations
df['year'] = df['year'].astype(int)
# Filter data for years 2000 to 2005
filtered_df = df[df['year'].between(2000, 2005)]
# Extract property taxes for these years
property_taxes = filtered_df['property taxes'].values
# Compute annual increases
increases = [property_taxes[i] - property_taxes[i-1] for i in range(1, len(property_taxes))]
# Calculate average increase
average_increase = sum(increases) / len(increases)
print(f"Final Answer: {average_increase:.0f}")