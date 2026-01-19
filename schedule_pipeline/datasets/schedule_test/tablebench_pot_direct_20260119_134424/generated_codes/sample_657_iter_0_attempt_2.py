import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'DC' and 'PCI' columns to numeric
df['DC'] = pd.to_numeric(df['DC'])
df['PCI'] = pd.to_numeric(df['PCI'])

# Calculate correlation coefficient between 'DC' and 'PCI'
correlation = df['DC'].corr(df['PCI'])

print(f"Final Answer: {correlation:.3f}")