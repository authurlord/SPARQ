import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['DC'] = pd.to_numeric(df['DC'])
df['PSDI'] = pd.to_numeric(df['PSDI'])
df['PSI'] = pd.to_numeric(df['PSI'])
df['PCI'] = pd.to_numeric(df['PCI'])

# Calculate correlation between DC and each factor
correlations = {
    'PSDI': df['DC'].corr(df['PSDI']),
    'PSI': df['DC'].corr(df['PSI']),
    'PCI': df['DC'].corr(df['PCI'])
}

# Check if any correlation is strong (|r| > 0.7)
significant_factors = [factor for factor, corr in correlations.items() if abs(corr) > 0.7]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")