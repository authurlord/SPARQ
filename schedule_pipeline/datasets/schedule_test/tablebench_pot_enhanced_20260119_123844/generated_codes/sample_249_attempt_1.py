import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil and gas companies with sales >= 300 billion
oil_gas_high_sales = df[(df['industry'] == 'oil and gas') & (df['sales (billion )'].astype(float) >= 300)]
# Calculate average market value
avg_market_value = oil_gas_high_sales['market value (billion )'].astype(float).mean()
print(f"Final Answer: {avg_market_value:.1f}")