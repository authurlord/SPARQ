import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation analysis
numerical_columns = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
correlation_matrix = df[numerical_columns].corr()

# Extract correlation of each feature with 'points'
correlations_with_points = correlation_matrix['points'].drop('points')

# Find features with significant correlation (absolute value > 0.5)
significant_features = correlations_with_points[abs(correlations_with_points) > 0.5].index.tolist()

if significant_features:
    print(f"Final Answer: {', '.join(significant_features)}")
else:
    print("Final Answer: no clear impact")