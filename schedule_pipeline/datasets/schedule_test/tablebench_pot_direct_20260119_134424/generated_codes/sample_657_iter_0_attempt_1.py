import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the Veneto region
veneto_data = df[df['Province'] == 'Veneto']
# Convert 'DC' and 'PCI' to numeric
veneto_data['DC'] = pd.to_numeric(veneto_data['DC'])
veneto_data['PCI'] = pd.to_numeric(veneto_data['PCI'])
# Calculate correlation coefficient
correlation = veneto_data['DC'].corr(veneto_data['PCI'])
print(f"Final Answer: {correlation:.3f}")