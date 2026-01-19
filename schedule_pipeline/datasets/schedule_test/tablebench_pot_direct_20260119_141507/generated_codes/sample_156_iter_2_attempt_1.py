import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns
columns_to_check = ['area km 2', 'area % of eu', 'pop density people / km 2']
target_column = 'population % of eu'

# Compute correlation with target column
correlations = df[columns_to_check].corrwith(df[target_column])

# Check if any correlation has an absolute value greater than 0.3 (a reasonable threshold for significance)
significant_correlations = correlations[abs(correlations) > 0.3]

if significant_correlations.empty:
    print("Final Answer: no clear impact")
else:
    # Return the names of the factors with significant correlation
    significant_factors = list(significant_correlations.index)
    print(f"Final Answer: {', '.join(significant_factors)}")