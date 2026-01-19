import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Total terrestrial vertebrates' to integer type
df['Total terrestrial vertebrates'] = pd.to_numeric(df['Total terrestrial vertebrates'])

# Find the country with the highest total
max_country = df.loc[df['Total terrestrial vertebrates'].idxmax(), 'Country']
max_value = df['Total terrestrial vertebrates'].max()

# Find the country with the lowest total
min_country = df.loc[df['Total terrestrial vertebrates'].idxmin(), 'Country']
min_value = df['Total terrestrial vertebrates'].min()

# Calculate the difference
difference = max_value - min_value

print(f"Final Answer: {max_country}, {difference}")