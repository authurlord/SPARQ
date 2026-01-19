import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_df = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to float
female_df['Metabolic rate'] = pd.to_numeric(female_df['Metabolic rate'])
# Find max and min metabolic rates
max_rate = female_df['Metabolic rate'].max()
min_rate = female_df['Metabolic rate'].min()
# Calculate difference
difference = max_rate - min_rate
print(f"Final Answer: {difference}")