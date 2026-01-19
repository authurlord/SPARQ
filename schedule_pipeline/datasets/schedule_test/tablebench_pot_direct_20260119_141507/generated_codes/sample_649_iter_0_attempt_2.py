import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'un budget' from percentage string to decimal
df['un budget'] = df['un budget'].str.rstrip('%').astype(float) / 100.0

# Extract the two columns for correlation
un_budget = df['un budget']
international_trade = df['international trade (millions of usd) 2011']

# Calculate the correlation coefficient
correlation_coefficient = un_budget.corr(international_trade)

print(f"Final Answer: {correlation_coefficient:.3f}")