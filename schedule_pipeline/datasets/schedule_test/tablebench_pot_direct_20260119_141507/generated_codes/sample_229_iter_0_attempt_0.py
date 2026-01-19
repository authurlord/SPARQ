import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
df['lowest mark'] = pd.to_numeric(df['lowest mark'], errors='coerce')
df['highest mark'] = pd.to_numeric(df['highest mark'], errors='coerce')
df['% pass'] = pd.to_numeric(df['% pass'], errors='coerce')

# Compute correlation with % pass
correlation_mean = df['mean'].corr(df['% pass'])
correlation_lowest = df['lowest mark'].corr(df['% pass'])
correlation_highest = df['highest mark'].corr(df['% pass'])

# Find the factor with the highest absolute correlation
max_corr = max(abs(correlation_mean), abs(correlation_lowest), abs(correlation_highest))
if max_corr == abs(correlation_mean):
    result = 'mean'
elif max_corr == abs(correlation_lowest):
    result = 'lowest mark'
else:
    result = 'highest mark'

print(f"Final Answer: {result}")