import pandas as pd

df = pd.read_csv('table.csv')
# Sum the population of the top 5 cities (first 5 rows)
total_population = df.iloc[:5]['population'].sum()
print(f"Final Answer: {total_population}")