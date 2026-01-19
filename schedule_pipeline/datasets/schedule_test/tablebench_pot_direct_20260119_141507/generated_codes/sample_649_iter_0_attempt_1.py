import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'un budget' from percentage string to float (remove '%')
df['un budget'] = df['un budget'].str.replace('%', '').astype(float)

# Extract the relevant columns
un_budget = df['un budget']
international_trade = df['international trade (millions of usd) 2011'].astype(float)

# Compute correlation coefficient
correlation_coefficient = un_budget.corr(international_trade)

print(f"Final Answer: {correlation_coefficient:.3f}")