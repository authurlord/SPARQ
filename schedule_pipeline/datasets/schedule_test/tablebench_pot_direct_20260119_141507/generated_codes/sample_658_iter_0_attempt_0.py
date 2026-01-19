import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
total_pop = df['total']
density = df['population density (per km square)']

# Calculate the correlation coefficient
correlation_coefficient = total_pop.corr(density)
print(f"Final Answer: {correlation_coefficient:.3f}")