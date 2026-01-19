import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the trailer counts for 1923 and 1927
# Filter rows for year 1923 and 1927
year_1923 = df[df['year'] == '1923']
year_1927 = df[df['year'] == '1927']

# Sum the 'trailers' column for each year
total_trailers_1923 = year_1923['trailers'].astype(str).str.replace('-', '0').astype(int).sum()
total_trailers_1927 = year_1927['trailers'].astype(str).str.replace('-', '0').astype(int).sum()

# Calculate percentage increase
percentage_increase = ((total_trailers_1927 - total_trailers_1923) / total_trailers_1923) * 100

print(f"Final Answer: {percentage_increase:.2f}")