import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_animals = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to float and calculate difference between max and min
max_rate = female_animals['Metabolic rate'].astype(float).max()
min_rate = female_animals['Metabolic rate'].astype(float).min()
difference = max_rate - min_rate
print(f"Final Answer: {difference:.1f}")