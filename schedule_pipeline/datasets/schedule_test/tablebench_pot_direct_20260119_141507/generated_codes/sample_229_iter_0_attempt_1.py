import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['% pass'] = pd.to_numeric(df['% pass'], errors='coerce')
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
df['lowest mark'] = pd.to_numeric(df['lowest mark'], errors='coerce')
df['highest mark'] = pd.to_numeric(df['highest mark'], errors='coerce')

# Calculate correlation with % pass
correlation_mean = df['% pass'].corr(df['mean'])
correlation_lowest = df['% pass'].corr(df['lowest mark'])
correlation_highest = df['% pass'].corr(df['highest mark'])

# Find the one with the highest absolute correlation
max_corr = max(abs(correlation_mean), abs(correlation_lowest), abs(correlation_highest))
if max_corr == abs(correlation_mean):
    result = 'mean'
elif max_corr == abs(correlation_lowest):
    result = 'lowest mark'
else:
    result = 'highest mark'

print(f"Final Answer: {result}")