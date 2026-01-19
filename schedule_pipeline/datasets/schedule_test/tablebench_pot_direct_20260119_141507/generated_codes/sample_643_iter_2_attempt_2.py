import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry companies
banking_companies = df[df['industry'] == 'banking']

# Extract the relevant columns
assets = banking_companies['assets (billion)']
profits = banking_companies['profits (billion)']

# Calculate correlation coefficient
correlation = assets.corr(profits)

print(f"Final Answer: {correlation:.2f}")