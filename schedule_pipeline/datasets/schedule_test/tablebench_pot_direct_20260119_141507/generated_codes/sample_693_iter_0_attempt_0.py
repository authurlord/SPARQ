import pandas as pd

df = pd.read_csv('table.csv')
# Filter only female records
female_data = df[df['Sex'] == 'Female']
# Find the row with the highest metabolic rate
max_metabolic_rate_row = female_data.loc[female_data['Metabolic rate'].idxmax()]
# Return the animal (index) with the highest metabolic rate
animal_with_highest_rate = max_metabolic_rate_row['Animal']
print(f"Final Answer: {animal_with_highest_rate}")