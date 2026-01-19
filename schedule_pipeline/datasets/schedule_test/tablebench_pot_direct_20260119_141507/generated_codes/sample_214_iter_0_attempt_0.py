import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert all relevant columns to numeric (in case of string formatting)
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['profits (billion )'] = pd.to_numeric(df['profits (billion )'], errors='coerce')
df['assets (billion )'] = pd.to_numeric(df['assets (billion )'], errors='coerce')
df['market value (billion )'] = pd.to_numeric(df['market value (billion )'], errors='coerce')

# Calculate correlation with market value
correlation_sales = df['sales (billion )'].corr(df['market value (billion )'])
correlation_profits = df['profits (billion )'].corr(df['market value (billion )'])
correlation_assets = df['assets (billion )'].corr(df['market value (billion )'])

# Find the strongest correlation
max_corr = max(correlation_sales, correlation_profits, correlation_assets)
if max_corr == correlation_sales:
    primary_driver = 'sales (billion)'
elif max_corr == correlation_profits:
    primary_driver = 'profits (billion)'
else:
    primary_driver = 'assets (billion)'

print(f"Final Answer: {primary_driver}")