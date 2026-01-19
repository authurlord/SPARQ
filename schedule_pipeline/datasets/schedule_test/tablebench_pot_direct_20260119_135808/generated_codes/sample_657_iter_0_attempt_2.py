import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the Veneto region
veneto_data = df[df['Province'] == 'Veneto']
# Convert 'DC' and 'PCI' to float
veneto_data['DC'] = veneto_data['DC'].astype(float)
veneto_data['PCI'] = veneto_data['PCI'].astype(float)
# Calculate correlation coefficient
correlation = veneto_data['DC'].corr(veneto_data['PCI'])
print(f"Final Answer: {correlation:.3f}")