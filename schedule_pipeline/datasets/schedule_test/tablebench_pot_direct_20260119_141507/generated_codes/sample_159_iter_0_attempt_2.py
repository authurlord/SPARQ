import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns
columns_to_check = ['no sat', 'highest mark', 'lowest mark', 'mean']
target = '% pass'

# Compute correlation with '% pass'
correlations = df[columns_to_check].corrwith(df[target]).abs()

# Find the factor with the highest correlation
top_factor = correlations.idxmax()
correlation_value = correlations[top_factor]

# Check if the correlation is strong (above 0.7 in absolute value)
if correlation_value >= 0.7:
    print(f"Final Answer: {top_factor}")
else:
    print("Final Answer: no clear impact")