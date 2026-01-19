import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['no sat'] = pd.to_numeric(df['no sat'])
df['% pass'] = pd.to_numeric(df['% pass'])
df['highest mark'] = pd.to_numeric(df['highest mark'])
df['lowest mark'] = pd.to_numeric(df['lowest mark'])
df['mean'] = pd.to_numeric(df['mean'])

# Calculate correlation with '% pass'
correlations = {
    'no sat': df['% pass'].corr(df['no sat']),
    'highest mark': df['% pass'].corr(df['highest mark']),
    'lowest mark': df['% pass'].corr(df['lowest mark']),
    'mean': df['% pass'].corr(df['mean'])
}

# Check for strong correlations
strong_corr = [k for k, v in correlations.items() if abs(v) > 0.7]

if strong_corr:
    print(f"Final Answer: {', '.join(strong_corr)}")
else:
    print("Final Answer: no clear impact")