import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for 1923 and 1927
trailers_1923 = df[df['year'] == '1923']['trailers'].sum()
trailers_1927 = df[df['year'] == '1927']['trailers'].sum()

# Calculate percentage increase
percentage_increase = ((trailers_1927 - trailers_1923) / trailers_1923) * 100
print(f"Final Answer: {percentage_increase:.1f}")