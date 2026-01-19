import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'draw' and 'points' to numeric
df['draw'] = pd.to_numeric(df['draw'])
df['points'] = pd.to_numeric(df['points'])
# Calculate correlation coefficient
correlation = df['draw'].corr(df['points'])
print(f"Final Answer: {correlation:.3f}")