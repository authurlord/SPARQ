import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for female animals (excluding the summary row)
female_data = df[(df['Sex'] == 'Female') & (df['Animal'].notna())]
# Extract metabolic rates
metabolic_rates = female_data['Metabolic rate'].astype(float)
max_rate = metabolic_rates.max()
min_rate = metabolic_rates.min()
difference = max_rate - min_rate
print(f"Final Answer: {difference:.2f}")