import pandas as pd

df = pd.read_csv('table.csv')

# Compute correlation between each factor and 'average'
correlations = df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']].corr()['average'].abs()

# Identify factors with correlation > 0.3 (considered significant)
significant_factors = correlations[correlations > 0.3].index.tolist()

if not significant_factors:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {', '.join(significant_factors)}")