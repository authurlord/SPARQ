import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 1923 and 1927
year_1923 = df[df['year'] == '1923']
year_1927 = df[df['year'] == '1927']

# Sum trailers for each year
trailers_1923 = year_1923['trailers'].fillna(0).astype(float).sum()
trailers_1927 = year_1927['trailers'].fillna(0).astype(float).sum()

# Calculate percentage increase
percentage_increase = ((trailers_1927 - trailers_1923) / trailers_1923) * 100

print(f"Final Answer: {percentage_increase:.1f}")