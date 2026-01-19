import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_animals = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to float
female_animals['Metabolic rate'] = pd.to_numeric(female_animals['Metabolic rate'])
# Calculate difference between highest and lowest metabolic rates
difference = female_animals['Metabolic rate'].max() - female_animals['Metabolic rate'].min()
print(f"Final Answer: {difference}")