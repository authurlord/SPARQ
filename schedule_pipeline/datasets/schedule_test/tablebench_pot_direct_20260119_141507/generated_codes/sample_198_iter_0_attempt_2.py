import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
gdp_per_capita = df['gdp per capita usd (2009 - 2011)']
pop_density = df['pop density ( / km square)']
total_gdp = df['gdp millions of usd (2009)']

# Calculate correlation with population density
corr_density = gdp_per_capita.corr(pop_density)

# Calculate correlation with total GDP
corr_total_gdp = gdp_per_capita.corr(total_gdp)

# Print the correlation values for comparison
print(f"Correlation with population density: {corr_density:.3f}")
print(f"Correlation with total GDP: {corr_total_gdp:.3f}")

# Determine which is stronger (by absolute value)
if abs(corr_density) > abs(corr_total_gdp):
    final_answer = "pop density"
else:
    final_answer = "total GDP"

print(f"Final Answer: {final_answer}")