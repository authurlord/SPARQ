import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_df = df[df['industry'] == 'banking']
# Clean and convert 'sales (billion )' to numeric, removing commas and handling errors
banking_df['sales (billion )'] = banking_df['sales (billion )'].str.replace(',', '').astype(float)
# Get top 3 sales and sum
total_sales = banking_df['sales (billion )'].nlargest(3).sum()
print(f"Final Answer: {total_sales:.2f}")