import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for Spanish-related language categories
spanish_rows = df[df['language'].isin(['spanish', 'only spanish', 'native and spanish'])]

# Sum the values across these rows for each municipality
spanish_population_per_municipality = spanish_rows.drop(columns=['language']).sum(axis=0)

# Find the municipality with the highest total
max_municipality = spanish_population_per_municipality.idxmax()
print(f"Final Answer: {max_municipality}")