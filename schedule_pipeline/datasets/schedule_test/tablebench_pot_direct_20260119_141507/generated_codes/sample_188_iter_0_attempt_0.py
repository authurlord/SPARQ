import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between sales and market value by industry
correlation_by_industry = df.groupby('industry')[['sales (billion )', 'market value (billion )']].corr()['market value (billion )']['sales (billion )']

print(f"Final Answer: {correlation_by_industry.to_dict()}")