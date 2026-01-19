import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest total terrestrial vertebrates
max_country = df.loc[df['Total terrestrial vertebrates'].idxmax(), 'Country']
max_value = df['Total terrestrial vertebrates'].max()
# Find the country with the lowest total terrestrial vertebrates
min_country = df.loc[df['Total terrestrial vertebrates'].idxmin(), 'Country']
min_value = df['Total terrestrial vertebrates'].min()
# Calculate the difference
difference = max_value - min_value
print(f"Final Answer: {max_country}, {difference}")