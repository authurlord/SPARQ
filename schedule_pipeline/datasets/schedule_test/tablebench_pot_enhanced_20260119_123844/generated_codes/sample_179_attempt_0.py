import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation analysis
numerical_columns = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
correlation_matrix = df[numerical_columns].corr()

# Extract correlation of 'points' with other variables
point_correlations = correlation_matrix['points'].drop('points')

# Check for significant correlations (absolute value > 0.5)
significant_factors = point_correlations[abs(point_correlations) > 0.5].index.tolist()

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")