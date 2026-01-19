import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for Spanish-related language types
spanish_rows = df[df['language'].isin(['spanish', 'only spanish', 'native and spanish'])]

# Since the columns are municipality names, we need to find the max across those rows
# We'll sum the values for each municipality across the Spanish-related rows
municipality_spans = spanish_rows.drop(columns=['language']).sum()

# Find the municipality with the highest total Spanish speakers
highest_municipality = municipality_spans.idxmax()
print(f"Final Answer: {highest_municipality}")