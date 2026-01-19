import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Sex is 'Female' and extract metabolic rates
female_metabolic_rates = df[df['Sex'] == 'Female']['Metabolic rate'].dropna()
# Calculate the difference between max and min
difference = female_metabolic_rates.max() - female_metabolic_rates.min()
print(f"Final Answer: {difference:.2f}")