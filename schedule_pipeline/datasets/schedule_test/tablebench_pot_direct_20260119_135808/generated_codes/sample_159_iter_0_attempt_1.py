import pandas as pd

df = pd.read_csv('table.csv')

# Convert necessary columns to numeric
df['no sat'] = pd.to_numeric(df['no sat'])
df['highest mark'] = pd.to_numeric(df['highest mark'])
df['lowest mark'] = pd.to_numeric(df['lowest mark'])
df['mean'] = pd.to_numeric(df['mean'])
df['% pass'] = pd.to_numeric(df['% pass'])

# Calculate correlations
correlations = {
    'no sat': df['% pass'].corr(df['no sat']),
    'highest mark': df['% pass'].corr(df['highest mark']),
    'lowest mark': df['% pass'].corr(df['lowest mark']),
    'mean': df['% pass'].corr(df['mean'])
}

# Find the factor with the highest absolute correlation
max_corr_factor = max(correlations, key=abs)
print(f"Final Answer: {max_corr_factor}")