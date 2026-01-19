import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
columns_to_convert = ['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric)

# Calculate correlation between each factor and average
correlations = df[columns_to_convert].corr()['average'].drop('average')

# Check for significant correlation (absolute value > 0.5)
significant_factors = correlations[abs(correlations) > 0.5].index.tolist()

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")