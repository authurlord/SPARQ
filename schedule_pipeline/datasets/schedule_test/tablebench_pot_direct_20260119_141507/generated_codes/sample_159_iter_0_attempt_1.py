import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric (handle any non-numeric issues)
df['no sat'] = pd.to_numeric(df['no sat'], errors='coerce')
df['no passed'] = pd.to_numeric(df['no passed'], errors='coerce')
df['% pass'] = pd.to_numeric(df['% pass'], errors='coerce')
df['highest mark'] = pd.to_numeric(df['highest mark'], errors='coerce')
df['lowest mark'] = pd.to_numeric(df['lowest mark'], errors='coerce')
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')

# Calculate correlation with '% pass'
correlations = {
    'no sat': df['no sat'].corr(df['% pass']),
    'no passed': df['no passed'].corr(df['% pass']),
    'highest mark': df['highest mark'].corr(df['% pass']),
    'lowest mark': df['lowest mark'].corr(df['% pass']),
    'mean': df['mean'].corr(df['% pass'])
}

# Find the factor with the highest absolute correlation
max_corr = max(abs(correlations[factor]) for factor in correlations)
best_factor = None
for factor, corr in correlations.items():
    if abs(corr) == max_corr:
        best_factor = factor
        break

if max_corr < 0.3:  # Threshold for meaningful correlation
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {best_factor}")