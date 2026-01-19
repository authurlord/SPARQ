import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['no sat'] = pd.to_numeric(df['no sat'])
df['highest mark'] = pd.to_numeric(df['highest mark'])
df['lowest mark'] = pd.to_numeric(df['lowest mark'])
df['mean'] = pd.to_numeric(df['mean'])
df['% pass'] = pd.to_numeric(df['% pass'])

# Calculate correlation with '% pass'
correlations = {
    'no sat': df['% pass'].corr(df['no sat']),
    'highest mark': df['% pass'].corr(df['highest mark']),
    'lowest mark': df['% pass'].corr(df['lowest mark']),
    'mean': df['% pass'].corr(df['mean'])
}

# Identify the factor with highest absolute correlation
max_corr_factor = max(correlations, key=correlations.get)
max_corr_value = correlations[max_corr_factor]

# Output result based on threshold
if abs(max_corr_value) >= 0.7:
    print(f"Final Answer: {max_corr_factor}")
else:
    print("Final Answer: no clear impact")