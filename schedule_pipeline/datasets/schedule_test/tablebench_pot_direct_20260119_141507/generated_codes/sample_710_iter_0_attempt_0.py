import pandas as pd

df = pd.read_csv('table.csv')
# Find the country with the highest and lowest total terrestrial vertebrates
max_row = df.loc[df['Total terrestrial vertebrates'].idxmax()]
min_row = df.loc[df['Total terrestrial vertebrates'].idxmin()]

highest_country = max_row['Country']
lowest_country = min_row['Country']
difference = max_row['Total terrestrial vertebrates'] - min_row['Total terrestrial vertebrates']

print(f"Final Answer: {highest_country}, {lowest_country}, {difference}")