import pandas as pd

df = pd.read_csv('table.csv')

# Convert all columns to numeric, handling any potential parsing issues
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values after conversion
df = df.dropna()

# Compute correlation between 'DC' and each factor
correlations = df[['DC', 'PSDI', 'PSI', 'PCI']].corr()['DC'].abs()

# Find factors with correlation greater than 0.3 (threshold for significant influence)
significant_factors = correlations[correlations > 0.3].index.tolist()

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")