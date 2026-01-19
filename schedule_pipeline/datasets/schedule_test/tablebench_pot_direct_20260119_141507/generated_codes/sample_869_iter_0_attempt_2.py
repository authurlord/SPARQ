import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Sex is 'Female' and exclude the summary row
female_data = df[df['Sex'] == 'Female'].dropna(subset=['Metabolic rate'])
# Extract metabolic rates as floats
metabolic_rates = female_data['Metabolic rate'].astype(float)
# Calculate difference between max and min
difference = metabolic_rates.max() - metabolic_rates.min()
print(f"Final Answer: {difference:.2f}")