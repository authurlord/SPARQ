import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas'].copy()
# Sort by rank to get top 5
oil_gas_df = oil_gas_df.sort_values(by='rank').head(5)
# Calculate average market value
avg_market_value = oil_gas_df['market value (billion )'].mean()
print(f"Final Answer: {avg_market_value:.1f}")