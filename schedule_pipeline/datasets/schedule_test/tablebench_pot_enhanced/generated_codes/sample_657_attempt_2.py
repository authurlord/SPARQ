import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the aggregate row 'Veneto' to consider only individual provinces
provinces = df[df['Province'] != 'Veneto']
# Convert 'DC' and 'PCI' to numeric
provinces['DC'] = pd.to_numeric(provinces['DC'])
provinces['PCI'] = pd.to_numeric(provinces['PCI'])
# Calculate correlation coefficient
correlation = provinces['DC'].corr(provinces['PCI'])
print(f"Final Answer: {correlation:.3f}")