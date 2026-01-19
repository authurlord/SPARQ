import pandas as pd

df = pd.read_csv('table.csv')

# Convert sales column to numeric, coercing errors to NaN
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')

# Filter companies in oil and gas industry with sales >= 300 billion
oil_gas_filter = df['industry'] == 'oil and gas'
sales_filter = df['sales (billion )'] >= 300
filtered_df = df[oil_gas_filter & sales_filter]

# Calculate average market value of filtered companies
if not filtered_df.empty:
    avg_market_value = filtered_df['market value (billion )'].mean()
else:
    avg_market_value = 0

print(f"Final Answer: {avg_market_value:.1f}")