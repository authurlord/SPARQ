import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'spanish' column for each municipality
spanish_population = df['spanish'].astype(int).sum()
# Find the municipality with the highest Spanish speaker population
municipality = df.iloc[:, 1:].sum(axis=0).idxmax()
print(f"Final Answer: {municipality}")