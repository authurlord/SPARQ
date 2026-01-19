import pandas as pd

df = pd.read_csv('table.csv')
# Filter companies in the 'oil and gas' industry
oil_gas_df = df[df['industry'] == 'oil and gas']
# Take the top 5 (first 5 rows) in the oil and gas industry
top_5_oil_gas = oil_gas_df.head(5)
# Calculate the average market value
avg_market_value = top_5_oil_gas['market value (billion )'].mean()
print(f"Final Answer: {avg_market_value:.1f}")