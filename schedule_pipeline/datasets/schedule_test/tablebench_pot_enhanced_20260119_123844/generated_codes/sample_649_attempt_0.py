import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'un budget' to float by removing '%' and converting to float
df['un budget'] = df['un budget'].str.rstrip('%').astype(float)
# Ensure 'international trade (millions of usd) 2011' is numeric
df['international trade (millions of usd) 2011'] = pd.to_numeric(df['international trade (millions of usd) 2011'])
# Calculate the correlation coefficient
correlation = df['un budget'].corr(df['international trade (millions of usd) 2011'])
print(f"Final Answer: {correlation:.4f}")