import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']] = df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']].apply(pd.to_numeric)

# Calculate correlation between each factor and average
correlations = df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']].corr()['average'].drop('average')

# Check for significant correlations (absolute value > 0.5)
significant_factors = correlations[abs(correlations) > 0.5].index.tolist()

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")