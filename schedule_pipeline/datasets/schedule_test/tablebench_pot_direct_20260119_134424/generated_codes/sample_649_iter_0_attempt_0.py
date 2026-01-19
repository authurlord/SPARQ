import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'un budget' from percentage string to float
df['un budget'] = df['un budget'].str.rstrip('%').astype(float)
# Convert 'international trade (millions of usd) 2011' to numeric
df['international trade (millions of usd) 2011'] = pd.to_numeric(df['international trade (millions of usd) 2011'])
# Calculate correlation coefficient
correlation = df['un budget'].corr(df['international trade (millions of usd) 2011'])
print(f"Final Answer: {correlation:.4f}")