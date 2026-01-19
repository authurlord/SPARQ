import pandas as pd

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation analysis
numeric_columns = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
df_numeric = df[numeric_columns]

# Compute correlation matrix
correlation_matrix = df_numeric.corr()

# Extract correlation of each feature with 'points'
correlations_with_points = correlation_matrix['points'].abs().sort_values(ascending=False)

# Identify features with significant correlation (> 0.5)
significant_features = correlations_with_points[correlations_with_points > 0.5].index.tolist()

# Remove 'points' itself from the list
significant_features = [feat for feat in significant_features if feat != 'points']

if significant_features:
    print(f"Final Answer: {', '.join(significant_features)}")
else:
    print("Final Answer: no clear impact")