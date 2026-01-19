import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['DC'] = pd.to_numeric(df['DC'])
df['PSDI'] = pd.to_numeric(df['PSDI'])
df['PSI'] = pd.to_numeric(df['PSI'])
df['PCI'] = pd.to_numeric(df['PCI'])

# Compute correlation matrix
correlation_matrix = df[['DC', 'PSDI', 'PSI', 'PCI']].corr()

# Extract correlations between DC and other factors
corr_psd = abs(correlation_matrix.loc['DC', 'PSDI'])
corr_psi = abs(correlation_matrix.loc['DC', 'PSI'])
corr_pci = abs(correlation_matrix.loc['DC', 'PCI'])

# Check if any correlation is strong (above 0.7)
if corr_psd > 0.7 or corr_psi > 0.7 or corr_pci > 0.7:
    if corr_psd > corr_psi and corr_psd > corr_pci:
        print("Final Answer: PSDI")
    elif corr_psi > corr_pci:
        print("Final Answer: PSI")
    else:
        print("Final Answer: PCI")
else:
    print("Final Answer: no clear impact")