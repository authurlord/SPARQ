import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['DC'] = pd.to_numeric(df['DC'])
df['PSDI'] = pd.to_numeric(df['PSDI'])
df['PSI'] = pd.to_numeric(df['PSI'])
df['PCI'] = pd.to_numeric(df['PCI'])

# Calculate correlation matrix
correlations = df[['DC', 'PSDI', 'PSI', 'PCI']].corr()['DC'].abs()

# Identify factors with significant influence (correlation > 0.5)
significant_factors = [col for col in correlations.index if col != 'DC' and correlations[col] > 0.5]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")