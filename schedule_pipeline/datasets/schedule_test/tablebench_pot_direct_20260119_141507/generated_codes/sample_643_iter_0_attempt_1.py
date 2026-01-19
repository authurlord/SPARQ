import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'banking'
banking_companies = df[df['industry'] == 'banking']

# Extract the 'assets (billion)' and 'profits (billion)' columns
assets = banking_companies['assets (billion)']
profits = banking_companies['profits (billion)']

# Calculate the correlation coefficient
correlation = assets.corr(profits)

print(f"Final Answer: {correlation:.2f}")