import pandas as pd

df = pd.read_csv('table.csv')

# Calculate correlation between 'DC' and each factor
correlations = df[['DC', 'PSDI', 'PSI', 'PCI']].corr()['DC'].abs()

# Find factors with correlation > 0.3
significant_factors = [col for col in correlations.index if correlations[col] > 0.3]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")