import pandas as pd

df = pd.read_csv('table.csv')
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include='number')

# Compute correlation with 'points'
correlations = numeric_df.corr()['points'].abs().sort_values(ascending=False)

# Check for significant correlations (threshold > 0.5)
significant_factors = [col for col, corr in correlations.items() if corr > 0.5 and col != 'points']

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")