import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Sex is 'Female' and exclude the row with all dashes
female_data = df[df['Sex'] == 'Female'].dropna(subset=['Metabolic rate'])
# Find the row with the highest metabolic rate
highest_metabolic_rate_row = female_data.loc[female_data['Metabolic rate'].idxmax()]
animal_with_highest_rate = highest_metabolic_rate_row['Animal']
print(f"Final Answer: {animal_with_highest_rate}")