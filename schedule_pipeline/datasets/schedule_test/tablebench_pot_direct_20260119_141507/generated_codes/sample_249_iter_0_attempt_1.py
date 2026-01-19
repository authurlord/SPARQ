import pandas as pd

df = pd.read_csv('table.csv')
# Filter companies in 'oil and gas' industry with sales >= 300 billion
filtered_df = df[(df['industry'] == 'oil and gas') & (df['sales (billion )'] >= 300)]
# Calculate average market value
avg_market_value = filtered_df['market value (billion )'].mean()
print(f"Final Answer: {avg_market_value:.1f}")