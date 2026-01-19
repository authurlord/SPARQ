import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas']
# Convert 'market value (billion )' to numeric, coercing errors to NaN
oil_gas_df['market value (billion )'] = pd.to_numeric(oil_gas_df['market value (billion )'], errors='coerce')
# Drop rows with invalid market values
oil_gas_df = oil_gas_df.dropna(subset=['market value (billion )'])
# Get top 5 by market value
top_5_market_value = oil_gas_df.nlargest(5, 'market value (billion )')['market value (billion )']
# Calculate average
average_market_value = top_5_market_value.mean()
print(f"Final Answer: {average_market_value:.1f}")