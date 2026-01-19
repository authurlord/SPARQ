import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'un budget' from percentage string to float
df['un budget'] = df['un budget'].str.rstrip('%').astype(float)
# Calculate correlation between 'un budget' and 'international trade (millions of usd) 2011'
correlation = df['un budget'].corr(df['international trade (millions of usd) 2011'].astype(float))
print(f"Final Answer: {correlation:.4f}")