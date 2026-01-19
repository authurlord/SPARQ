import pandas as pd

df = pd.read_csv('table.csv')

# Correctly handle column names with spaces and parentheses
df.columns = df.columns.str.strip()

# Calculate correlation between sales and market value by industry
correlation_by_industry = df.groupby('industry')[['sales (billion )', 'market value (billion )']].corr().iloc[:, 0].drop('sales (billion )')

# Output the correlation values (between sales and market value) for each industry
print(f"Final Answer: {correlation_by_industry.to_dict()}")