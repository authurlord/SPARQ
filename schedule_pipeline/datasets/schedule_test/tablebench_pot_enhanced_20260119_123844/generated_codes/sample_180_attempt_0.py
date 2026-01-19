import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['DC'] = pd.to_numeric(df['DC'])
df['PSDI'] = pd.to_numeric(df['PSDI'])
df['PSI'] = pd.to_numeric(df['PSI'])
df['PCI'] = pd.to_numeric(df['PCI'])

# Calculate correlation coefficients
corr_psd = df['DC'].corr(df['PSDI'])
corr_psi = df['DC'].corr(df['PSI'])
corr_pci = df['DC'].corr(df['PCI'])

# Check for significant influence (absolute value > 0.7)
if abs(corr_psd) > 0.7:
    result = "PSDI"
elif abs(corr_psi) > 0.7:
    result = "PSI"
elif abs(corr_pci) > 0.7:
    result = "PCI"
else:
    result = "no clear impact"

print(f"Final Answer: {result}")