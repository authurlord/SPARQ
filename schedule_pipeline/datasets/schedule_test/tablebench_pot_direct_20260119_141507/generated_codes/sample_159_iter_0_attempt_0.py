import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns for correlation analysis
columns_to_analyze = ['no sat', 'highest mark', 'lowest mark', 'mean']
target_column = '% pass'

# Compute correlation with % pass
correlations = df[columns_to_analyze].corrwith(df[target_column])

# Find the factor with the highest absolute correlation
max_corr = correlations.abs().idxmax()
max_corr_value = correlations.abs().max()

# If the maximum correlation is below a threshold (e.g., 0.3), consider no clear impact
if max_corr_value < 0.3:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {max_corr}")