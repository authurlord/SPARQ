import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['preliminaries'] = pd.to_numeric(df['preliminaries'])
df['interview'] = pd.to_numeric(df['interview'])
df['swimsuit'] = pd.to_numeric(df['swimsuit'])
df['evening gown'] = pd.to_numeric(df['evening gown'])
df['average'] = pd.to_numeric(df['average'])

# Calculate correlation between each factor and average
correlations = {
    'preliminaries': df['preliminaries'].corr(df['average']),
    'interview': df['interview'].corr(df['average']),
    'swimsuit': df['swimsuit'].corr(df['average']),
    'evening gown': df['evening gown'].corr(df['average'])
}

# Check for significant influence (absolute correlation > 0.5)
significant_factors = [factor for factor, corr in correlations.items() if abs(corr) > 0.5]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")