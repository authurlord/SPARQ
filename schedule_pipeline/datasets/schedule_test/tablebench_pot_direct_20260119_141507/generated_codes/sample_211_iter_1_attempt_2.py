import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
density = df['population density (per km square)']
growth_rate = df['annual population growth rate']

# Convert growth rate to numeric (handle percentage strings)
growth_rate_numeric = pd.to_numeric(growth_rate.str.rstrip('%'), errors='coerce')

# Compute correlation
correlation = density.corr(growth_rate_numeric)

print(f"Final Answer: correlation_{correlation:.3f}")