import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_df = df[df['industry'] == 'banking']
# Extract sales values and sum them
total_sales = banking_df['sales (billion )'].sum()
print(f"Final Answer: {total_sales}")