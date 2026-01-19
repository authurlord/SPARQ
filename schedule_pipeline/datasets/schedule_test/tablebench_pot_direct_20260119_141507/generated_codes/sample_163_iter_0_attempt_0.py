import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns
sales = df['sales (billion )']
profits = df['profits (billion )']
assets = df['assets (billion )']
market_value = df['market value (billion )']

# Calculate correlation with market value
correlation_sales = sales.corr(market_value)
correlation_profits = profits.corr(market_value)
correlation_assets = assets.corr(market_value)

# Find the factor with the highest absolute correlation
max_corr = max(correlation_sales, correlation_profits, correlation_assets, key=abs)
if max_corr == correlation_sales:
    final_factor = 'sales (billion )'
elif max_corr == correlation_profits:
    final_factor = 'profits (billion )'
else:
    final_factor = 'assets (billion )'

print(f"Final Answer: {final_factor}")