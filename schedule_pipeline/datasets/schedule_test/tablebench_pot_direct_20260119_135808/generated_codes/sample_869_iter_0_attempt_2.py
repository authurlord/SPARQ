import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_animals = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to float
female_animals['Metabolic rate'] = pd.to_numeric(female_animals['Metabolic rate'])
# Find max and min metabolic rates
max_rate = female_animals['Metabolic rate'].max()
min_rate = female_animals['Metabolic rate'].min()
# Calculate difference
difference = max_rate - min_rate
print(f"Final Answer: {difference:.1f}")