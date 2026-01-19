import pandas as pd

df = pd.read_csv('table.csv')
# Filter banking industry companies
banking_companies = df[df['industry'] == 'banking']

# Select the relevant columns, trimming any extra spaces
assets = banking_companies['assets (billion )'].astype(float)
profits = banking_companies['profits (billion )'].astype(float)

# Calculate correlation coefficient
correlation = assets.corr(profits)

print(f"Final Answer: {correlation:.2f}")