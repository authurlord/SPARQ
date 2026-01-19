import pandas as pd

df = pd.read_csv('table.csv')

# Correctly access columns with spaces using quotes
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Group by industry and compute correlation between sales and market value
correlations = df.groupby('industry')[['sales (billion )', 'market value (billion )']].corr().iloc[:, 1].drop('sales (billion )')

print(f"Final Answer: {correlations.to_dict()}")