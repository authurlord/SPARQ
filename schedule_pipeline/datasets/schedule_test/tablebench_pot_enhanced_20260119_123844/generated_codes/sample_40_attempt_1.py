import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'population (2010 census)' column
total_population = df['population (2010 census)'].sum()
print(f"Final Answer: {total_population}")