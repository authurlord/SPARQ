import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas']
# Filter for sales >= 300 billion
high_sales_df = oil_gas_df[oil_gas_df['sales (billion )'].astype(float) >= 300]
# Calculate average market value
avg_market_value = high_sales_df['market value (billion )'].astype(float).mean()
print(f"Final Answer: {avg_market_value:.1f}")