import pandas as pd

df = pd.read_csv('table.csv')
# Convert sales and market value to numeric
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['sales (billion )', 'market value (billion )'])

# Group by industry and compute correlation between sales and market value
correlations = df.groupby('industry')[['sales (billion )', 'market value (billion )']].corr()['market value (billion )']['sales (billion )']

print(f"Final Answer: {correlations.to_dict()}")