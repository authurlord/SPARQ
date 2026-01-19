import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 cities by rank (1st to 5th)
top_5_population = df.iloc[:5]['population'].sum()
print(f"Final Answer: {top_5_population}")