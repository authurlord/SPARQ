import pandas as pd

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation
numeric_cols = ['area km 2', 'area % of eu', 'pop density people / km 2', 'population % of eu']
df_numeric = df[numeric_cols].dropna()

# Compute correlation with 'population % of eu'
correlations = df_numeric.corr()['population % of eu'].abs()

# Find if any correlation exceeds 0.3 (considered significant)
significant_factors = correlations[correlations > 0.3].index.tolist()

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")