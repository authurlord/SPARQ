import pandas as pd

df = pd.read_csv('table.csv')
# Correctly access columns with spaces by using double quotes
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Compute correlation between sales and market value by industry
correlations = df.groupby('industry')[['sales (billion )', 'market value (billion )']].corr()['market value (billion )']['sales (billion )']

print(f"Final Answer: {correlations.to_dict()}")