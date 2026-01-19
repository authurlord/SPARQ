import pandas as pd

df = pd.read_csv('table.csv')
# Convert population column to integer and sum
total_population = df['population (2010 census)'].astype(int).sum()
print(f"Final Answer: {total_population}")