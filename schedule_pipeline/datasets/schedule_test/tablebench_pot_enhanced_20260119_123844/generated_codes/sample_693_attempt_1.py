import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_animals = df[df['Sex'] == 'Female']
# Find the animal with the highest metabolic rate
max_metabolic_rate_animal = female_animals.loc[female_animals['Metabolic rate'].idxmax(), 'Animal']
print(f"Final Answer: {max_metabolic_rate_animal}")