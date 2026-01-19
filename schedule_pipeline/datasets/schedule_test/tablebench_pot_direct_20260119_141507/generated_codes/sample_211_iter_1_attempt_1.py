import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns
density = df['population density (per km square)']
growth_rate = df['annual population growth rate']

# Calculate Pearson correlation coefficient
correlation = density.corr(growth_rate)

print(f"Final Answer: {correlation:.3f}")