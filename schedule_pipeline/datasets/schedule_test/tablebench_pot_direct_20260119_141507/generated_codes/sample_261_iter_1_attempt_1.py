import pandas as pd

df = pd.read_csv('table.csv')

# Filter companies in the oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas'].copy()

# Sort by market value in descending order and take top 5
top_5_oil_gas = oil_gas_df.sort_values(by='market value (billion )', ascending=False).head(5)

# Calculate the average market value of the top 5
avg_market_value = top_5_oil_gas['market value (billion )'].mean()

print(f"Final Answer: {avg_market_value:.1f}")