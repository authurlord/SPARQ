import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil and gas industry
oil_gas_companies = df[df['industry'] == 'oil and gas']
# Sort by rank to get the top 5
top_5_oil_gas = oil_gas_companies.sort_values('rank').head(5)
# Calculate average market value
avg_market_value = top_5_oil_gas['market value (billion )'].astype(float).mean()
print(f"Final Answer: {avg_market_value:.1f}")