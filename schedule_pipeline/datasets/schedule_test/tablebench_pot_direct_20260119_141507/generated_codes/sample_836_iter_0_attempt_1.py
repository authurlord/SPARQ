import pandas as pd

df = pd.read_csv('table.csv')
# Find the municipality with the highest Spanish speaker population
max_spanish_municipality = df.loc[df['spanish'].idxmax(), 'language']
print(f"Final Answer: {max_spanish_municipality}")