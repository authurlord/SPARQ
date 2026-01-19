import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Total terrestrial vertebrates' to numeric, coercing errors to NaN if any
df['Total terrestrial vertebrates'] = pd.to_numeric(df['Total terrestrial vertebrates'], errors='coerce')

# Find the country with the highest and lowest total terrestrial vertebrates
max_country = df.loc[df['Total terrestrial vertebrates'].idxmax(), 'Country']
min_country = df.loc[df['Total terrestrial vertebrates'].idxmin(), 'Country']
max_value = df['Total terrestrial vertebrates'].max()
min_value = df['Total terrestrial vertebrates'].min()
difference = max_value - min_value

print(f"Final Answer: {max_country}, {difference}")