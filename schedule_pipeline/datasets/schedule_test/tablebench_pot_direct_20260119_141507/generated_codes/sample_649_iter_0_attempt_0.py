import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'un budget' from percentage string to float
df['un budget'] = df['un budget'].str.rstrip('%').astype(float)

# Extract the relevant columns
un_budget = df['un budget']
international_trade = df['international trade (millions of usd) 2011'].astype(float)

# Calculate the correlation coefficient
correlation = un_budget.corr(international_trade)

print(f"Final Answer: {correlation:.3f}")