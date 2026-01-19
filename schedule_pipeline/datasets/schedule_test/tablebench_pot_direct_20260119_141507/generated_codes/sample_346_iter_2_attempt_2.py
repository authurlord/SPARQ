import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where language is 'spanish'
spanish_row = df[df['language'] == 'spanish']
# Extract the values for each municipality (all columns except 'language')
municipalities = spanish_row.iloc[:, 1:].values.flatten()
# Count how many are >= 40000
count = sum(1 for pop in municipalities if pop >= 40000)
print(f"Final Answer: {count}")