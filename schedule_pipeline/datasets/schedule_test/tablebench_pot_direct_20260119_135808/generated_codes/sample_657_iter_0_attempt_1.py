import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the aggregated 'Veneto' row
provinces_data = df[df['Province'] != 'Veneto']
# Convert 'DC' and 'PCI' to numeric
provinces_data['DC'] = pd.to_numeric(provinces_data['DC'])
provinces_data['PCI'] = pd.to_numeric(provinces_data['PCI'])
# Calculate correlation coefficient
correlation = provinces_data['DC'].corr(provinces_data['PCI'])
print(f"Final Answer: {correlation:.3f}")