import pandas as pd

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation
numeric_cols = ['area (km square)', 'population density', 'change (%)']
df_numeric = df[numeric_cols].dropna()

# Compute correlation between 'change (%)' and other variables
correlations = df_numeric.corr()['change (%)'].abs().sort_values(ascending=False)

# Check if any correlation is above a reasonable threshold (e.g., 0.3)
significant_correlations = correlations[correlations > 0.3]

if significant_correlations.empty:
    print("Final Answer: no clear impact")
else:
    # Return the factor(s) with significant correlation
    significant_factors = ", ".join(significant_correlations.index.tolist())
    print(f"Final Answer: {significant_factors}")