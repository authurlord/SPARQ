import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Sex is Female and exclude the header row
female_data = df[df['Sex'] == 'Female']
# Find the animal with the highest metabolic rate
max_metabolic_rate_row = female_data.loc[female_data['Metabolic rate'].idxmax()]
animal_with_highest_rate = max_metabolic_rate_row['Animal']
print(f"Final Answer: {animal_with_highest_rate}")