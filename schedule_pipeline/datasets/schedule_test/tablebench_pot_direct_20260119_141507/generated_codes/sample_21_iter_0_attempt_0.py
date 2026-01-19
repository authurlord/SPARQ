import pandas as pd

df = pd.read_csv('table.csv')
# Select the top 5 cities by rank and sum their populations
total_population = df.head(5)['population'].sum()
print(f"Final Answer: {total_population}")