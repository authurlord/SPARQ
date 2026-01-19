import pandas as pd

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation
numeric_columns = ['area (km square)', 'population density', 'population (2011)', 'population (2006)']
df_numeric = df[numeric_columns].dropna()

# Calculate correlation with 'change (%)'
change_col = 'change (%)'
correlations = df_numeric.corrwith(df[change_col]).abs()

# Find columns with correlation greater than 0.3
significant_factors = correlations[correlations > 0.3].index.tolist()

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")