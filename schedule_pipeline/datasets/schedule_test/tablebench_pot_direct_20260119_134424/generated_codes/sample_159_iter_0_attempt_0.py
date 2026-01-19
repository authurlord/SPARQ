import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['no sat'] = pd.to_numeric(df['no sat'])
df['no passed'] = pd.to_numeric(df['no passed'])
df['% pass'] = pd.to_numeric(df['% pass'])
df['highest mark'] = pd.to_numeric(df['highest mark'])
df['lowest mark'] = pd.to_numeric(df['lowest mark'])
df['mean'] = pd.to_numeric(df['mean'])

# Calculate correlation between '% pass' and each factor
corr_highest = df['% pass'].corr(df['highest mark'])
corr_lowest = df['% pass'].corr(df['lowest mark'])
corr_mean = df['% pass'].corr(df['mean'])

# Check for strong correlations
if abs(corr_mean) > 0.5:
    print("Final Answer: mean")
elif abs(corr_highest) > 0.5:
    print("Final Answer: highest mark")
elif abs(corr_lowest) > 0.5:
    print("Final Answer: lowest mark")
else:
    print("Final Answer: no clear impact")