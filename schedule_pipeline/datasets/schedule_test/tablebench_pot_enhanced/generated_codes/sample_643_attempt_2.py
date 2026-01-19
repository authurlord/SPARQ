import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry companies
banking_df = df[df['industry'] == 'banking']
# Convert columns to numeric
banking_df['assets (billion )'] = pd.to_numeric(banking_df['assets (billion )'])
banking_df['profits (billion )'] = pd.to_numeric(banking_df['profits (billion )'])
# Calculate correlation coefficient
correlation = banking_df['assets (billion )'].corr(banking_df['profits (billion )'])
print(f"Final Answer: {correlation:.3f}")