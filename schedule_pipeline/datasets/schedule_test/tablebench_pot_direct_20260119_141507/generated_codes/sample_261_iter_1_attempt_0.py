import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas']

# Ensure market value is numeric
oil_gas_df['market value (billion )'] = pd.to_numeric(oil_gas_df['market value (billion )'], errors='coerce')

# Take the top 5 companies (first 5 rows) in oil and gas industry
top_5_oil_gas = oil_gas_df.head(5)

# Calculate average market value
avg_market_value = top_5_oil_gas['market value (billion )'].mean()
print(f"Final Answer: {avg_market_value:.1f}")