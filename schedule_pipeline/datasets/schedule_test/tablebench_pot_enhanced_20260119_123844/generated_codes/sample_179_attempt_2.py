import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to numeric
numerical_columns = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Calculate correlation matrix for numerical columns
correlation_matrix = df[numerical_columns].corr()

# Extract correlation of each feature with 'points'
correlations_with_points = correlation_matrix['points'].drop('points')

# Check if any correlation is strong (absolute value > 0.5)
strong_correlations = correlations_with_points[abs(correlations_with_points) > 0.5]

if len(strong_correlations) == 0:
    print("Final Answer: no clear impact")
else:
    # Return the features with strong correlation
    significant_features = ', '.join(strong_correlations.index)
    print(f"Final Answer: {significant_features}")