import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest total terrestrial vertebrates
max_country = df.loc[df['Total terrestrial vertebrates'].idxmax(), 'Country']
min_country = df.loc[df['Total terrestrial vertebrates'].idxmin(), 'Country']
max_value = df['Total terrestrial vertebrates'].max()
min_value = df['Total terrestrial vertebrates'].min()
difference = max_value - min_value

print(f"Final Answer: {max_country}, {difference}")