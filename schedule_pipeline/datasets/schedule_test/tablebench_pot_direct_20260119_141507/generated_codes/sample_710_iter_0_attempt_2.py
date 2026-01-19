import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'Total terrestrial vertebrates' column
vertebrate_counts = df['Total terrestrial vertebrates']

# Find the country with the highest and lowest values
max_country = df.loc[vertebrate_counts.idxmax(), 'Country']
min_country = df.loc[vertebrate_counts.idxmin(), 'Country']
difference = vertebrate_counts.max() - vertebrate_counts.min()

print(f"Final Answer: {max_country}, {min_country}, {difference}")