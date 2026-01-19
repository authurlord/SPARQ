import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'population' column to integer
df['population'] = df['population'].astype(int)
# Sum the population of the top 5 cities
total_population = df['population'].head(5).sum()
print(f"Final Answer: {total_population}")