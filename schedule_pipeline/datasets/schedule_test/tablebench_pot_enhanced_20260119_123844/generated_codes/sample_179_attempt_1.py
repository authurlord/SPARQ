import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to numeric, handling any non-numeric values
numerical_columns = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in numerical columns
df.dropna(subset=numerical_columns, inplace=True)

# Compute correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Extract correlation of each factor with 'points'
correlations_with_points = correlation_matrix['points'].abs().sort_values(ascending=False)

# Define threshold for significant impact (e.g., correlation > 0.5)
significant_factors = correlations_with_points[correlations_with_points > 0.5].index.tolist()

# Remove 'points' itself from the list
significant_factors = [factor for factor in significant_factors if factor != 'points']

if not significant_factors:
    print("Final Answer: no clear impact")
else:
    # Output only the most impactful factors (as per question, list them)
    print(f"Final Answer: {', '.join(significant_factors)}")