import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the last row (Veneto aggregate) and focus on individual provinces
provinces_data = df.iloc[:-1]  # All rows except the last one
# Convert 'DC' and 'PCI' to float for correlation calculation
dc_values = provinces_data['DC'].astype(float)
pci_values = provinces_data['PCI'].astype(float)
# Calculate correlation coefficient
correlation = dc_values.corr(pci_values)
print(f"Final Answer: {correlation:.3f}")