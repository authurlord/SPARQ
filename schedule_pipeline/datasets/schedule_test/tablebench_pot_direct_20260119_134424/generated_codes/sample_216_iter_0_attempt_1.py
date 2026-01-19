import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'her age' and 'of children' to numeric
df['her age'] = pd.to_numeric(df['her age'])
df['of children'] = pd.to_numeric(df['of children'])

# Calculate correlation coefficient
correlation = df['her age'].corr(df['of children'])

print(f"Final Answer: {correlation:.2f}")