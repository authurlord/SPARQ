import pandas as pd

df = pd.read_csv('table.csv')

# Compute correlation between each factor and 'average'
correlations = df[['preliminaries', 'interview', 'swimsuit', 'evening gown', 'average']].corr()['average'].abs()

# Find the factor with the highest correlation
top_factor = correlations.idxmax()
correlation_value = correlations[top_factor]

# Determine if it's significant (threshold: > 0.3)
if correlation_value >= 0.3:
    print(f"Final Answer: {top_factor}")
else:
    print("Final Answer: no clear impact")