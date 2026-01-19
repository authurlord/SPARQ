import pandas as pd

df = pd.read_csv('table.csv')
# Convert the columns to numeric for correlation analysis
df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']] = df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']].apply(pd.to_numeric)

# Calculate correlation between each factor and the average score
correlations = df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']].corr()['average'].drop('average')

# Check for significant correlations (absolute value > 0.5)
significant_factors = [factor for factor, corr in correlations.items() if abs(corr) > 0.5]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")