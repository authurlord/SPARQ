import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'oil and gas'
oil_gas_companies = df[df['industry'] == 'oil and gas']
# Sort by rank to get the top 5
top_5_oil_gas = oil_gas_companies.nsmallest(5, 'rank')
# Calculate the average market value
avg_market_value = top_5_oil_gas['market value (billion )'].astype(float).mean()
print(f"Final Answer: {avg_market_value:.1f}")