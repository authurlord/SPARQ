import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for 1923 and 1927
year_1923 = df[df['year'] == '1923']
year_1927 = df[df['year'] == '1927']

# Sum the 'trailers' column for these years
total_trailers_1923 = year_1923['trailers'].sum()
total_trailers_1927 = year_1927['trailers'].sum()

# Calculate percentage increase
percentage_increase = ((total_trailers_1927 - total_trailers_1923) / total_trailers_1923) * 100

print(f"Final Answer: {percentage_increase:.2f}")