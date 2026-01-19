import pandas as pd

df = pd.read_csv('table.csv')
# Select the row where language is 'spanish'
spanish_row = df[df['language'] == 'spanish']
# Extract the values for each municipality
spanish_populations = spanish_row.iloc[:, 1:]  # All columns after 'language'
# Count how many are >= 40000
count = (spanish_populations >= 40000).sum().sum()
print(f"Final Answer: {count}")