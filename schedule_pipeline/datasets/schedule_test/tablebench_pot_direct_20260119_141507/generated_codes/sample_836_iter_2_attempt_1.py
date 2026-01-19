import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where language is 'spanish'
spanish_row = df[df['language'] == 'spanish']
# Get the index of that row and extract the municipality with the highest value
max_population_municipality = spanish_row.iloc[0].idxmax()
print(f"Final Answer: {max_population_municipality}")