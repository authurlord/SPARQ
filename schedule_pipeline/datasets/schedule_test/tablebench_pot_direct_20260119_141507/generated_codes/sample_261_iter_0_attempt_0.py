import pandas as pd

df = pd.read_csv('table.csv')
# Filter companies in the oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas']

# Take the first 5 companies (by row order)
top_5_oil_gas = oil_gas_df.head(5)

# Calculate average market value
average_market_value = top_5_oil_gas['market value (billion )'].mean()
print(f"Final Answer: {average_market_value:.1f}")