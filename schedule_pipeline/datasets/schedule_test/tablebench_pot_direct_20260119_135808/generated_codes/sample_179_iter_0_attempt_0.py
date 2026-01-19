import pandas as pd

df = pd.read_csv('table.csv')

# Select numeric columns related to performance and points
performance_cols = ['wins', 'poles', 'laps', 'podiums', 'points']

# Compute correlation matrix
correlation_matrix = df[performance_cols].corr()

# Extract correlation of each factor with points (excluding self-correlation)
correlations_with_points = correlation_matrix['points'].drop('points')

# Define threshold for significant correlation
threshold = 0.5
significant_factors = correlations_with_points[abs(correlations_with_points) > threshold]

if len(significant_factors) == 0:
    print("Final Answer: no clear impact")
else:
    # Return the names of significant factors
    print(f"Final Answer: {', '.join(significant_factors.index)}")