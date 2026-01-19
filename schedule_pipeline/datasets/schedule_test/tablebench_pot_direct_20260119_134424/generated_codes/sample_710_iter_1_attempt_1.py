import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Total terrestrial vertebrates' to integer type for numerical operations
df['Total terrestrial vertebrates'] = pd.to_numeric(df['Total terrestrial vertebrates'])

# Find the country with the highest and lowest 'Total terrestrial vertebrates'
max_country = df.loc[df['Total terrestrial vertebrates'].idxmax(), 'Country']
min_country = df.loc[df['Total terrestrial vertebrates'].idxmin(), 'Country']
max_value = df['Total terrestrial vertebrates'].max()
min_value = df['Total terrestrial vertebrates'].min()

# Calculate the difference
difference = max_value - min_value

print(f"Final Answer: {max_country}, {difference}")