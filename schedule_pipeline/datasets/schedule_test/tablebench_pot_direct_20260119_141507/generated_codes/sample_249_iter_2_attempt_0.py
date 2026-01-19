import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling any parsing issues
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Filter companies in 'oil and gas' industry with sales >= 300
oil_gas_sales_filtered = df[
    (df['industry'] == 'oil and gas') & 
    (df['sales (billion )'] >= 300)
]

# Calculate average market value of filtered companies
if not oil_gas_sales_filtered.empty:
    avg_market_value = oil_gas_sales_filtered['market value (billion )'].mean()
    print(f"Final Answer: {avg_market_value:.1f}")
else:
    print("Final Answer: 0")