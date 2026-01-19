import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns for correlation
numerical_cols = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
correlation_matrix = df[numerical_cols].corr()

# Extract correlations between each factor and 'points'
correlations = correlation_matrix['points'].abs().sort_values(ascending=False)

# Find factors with significant correlation (threshold > 0.3)
significant_factors = []
for col in ['races', 'wins', 'poles', 'laps', 'podiums']:
    if abs(correlations[col]) > 0.3:
        significant_factors.append(col)

if not significant_factors:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {', '.join(significant_factors)}")